import torch
import torch.nn as nn
from torchmetrics import TweedieDevianceScore, R2Score
from sklearn.model_selection import train_test_split, KFold
import copy
import os
import numpy as np
import collections
from load_data_as_tensor_v2 import load_data_as_tensor as load_data_as_tensor_v2
import time
total_foresight=20 #basically f=total_foresight/2 NOTE must be even number
s=4
with_thetas=0 #yes=1 no=0
with_normal_lenght=0 #difference between v1 and v2
input_size=(2+with_normal_lenght+with_thetas)*(total_foresight+1)
hidden_size1=450
hidden_size2and3=200
sampling=4
output_size=2*sampling+1
starting_learning_rate = 0.003 # learning rate
epochs = 50
number_of_models=1

class trackNet(nn.Module):
    def __init__(self,input_size, hidden_size1,hidden_size2and3, output_size):
        
        super().__init__()
        self.flatten = nn.Flatten()
        self.model_stack=nn.Sequential(
        nn.Linear(input_size,hidden_size1,dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(hidden_size1,hidden_size2and3,dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(hidden_size2and3,hidden_size2and3,dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(hidden_size2and3,output_size,dtype=torch.float64),
        nn.Hardsigmoid()
        )   

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.model_stack(x)
        return logits

def train(files, models, loss_fn):
    #il singolo filename è una batch di training data
    #se faccio questo vuol dire che in pratica lo stesso dato deve essere utilizzato come lable su neuroni diversi
    tot_time=0

    
    for batch, file in enumerate(files):
        for i in range(len(models["model"])):
            model=models["model"][i]
            model.train()

            optimizer=models["optimizer"][i]
            model.train()
            
            #print(batch)
            X,Y = file
            track_length=X.shape[0]
            #print(X.shape, Y.shape)

            t_batch=time.time()
            
            batch_X = X.view(track_length, -1)   # shape: (track_length, input_size * n_features)
            pred = model(batch_X)                # shape: (track_length, output_size)
            loss = loss_fn(pred, Y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tot_time+=time.time()-t_batch
            
            if batch % 500 == 0:
                loss, current = loss.item(), (batch + 1)
                r2_obj.update(pred, Y)
                r2=r2_obj.compute()
                r2_obj.reset()
                print(f"model {i+1} loss: {loss:>7f}, r2: {r2}  [{current:>5d}/{len(files):>5d}], mean time for exec: {tot_time/(batch+1)}")
    
    return tot_time   

def evaluation(filenames, models, loss_fn):
    min_loss=np.inf
    best_model=None
    best_lr=None
    for i in range(len(models["model"])):
        model=models["model"][i]
        
        lr=models["optimizer"][i].param_groups[0]["lr"]

        avg_loss,avg_r2,avg_poiss = test(filenames,model,loss_fn)
        #somehow find a way to incorporate other things in the decision
        if avg_loss<=min_loss:
            best_model=copy.deepcopy(model.state_dict())
            min_loss=avg_loss
            best_lr=lr
            print("new best model! ",avg_loss)
    for i in range(len(models["model"])):
        learning_rate= starting_learning_rate*(1+0.25*(i-int(len(models["model"])/2))) # ex with 4 models starting_learning_rate * [0.5,0.75,1,1.25,1.5]
        models["model"][i].load_state_dict(best_model)
        models["optimizer"][i].param_groups[0]["lr"]=learning_rate


    return best_model, min_loss

def test(files, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_r2 = 0.0
    total_poiss = 0.0
    with torch.no_grad():
        for file in files:
            # carica i dati e li porta sul device corretto
            X,Y = file

            # forward
            pred = model(X.view(X.shape[0], -1))  # appiattisce input se necessario

            # calcola loss
            loss = loss_fn(pred, Y)
            total_loss += loss.item()
            r2_obj.update(pred, Y)
            total_r2+=r2_obj.compute()
            r2_obj.reset()
            total_poiss+=poiss_obj(pred,Y)
            poiss_obj.reset()
            

    # media sulla lunghezza del test set
    avg_loss = total_loss / len(files)
    avg_r2=total_r2/len(files)
    avg_poiss=total_poiss/len(files)
    return avg_loss,avg_r2,avg_poiss

clock=time.time()
tracks_dir = "tracks/train/featureExtracted"
racing_line_dir = "tracks/train/racelinesCorrected"
#uso il singolo circuito come batch? si dai
filenames =np.array( [f for f in os.listdir(tracks_dir) if os.path.isfile(os.path.join(tracks_dir, f))])
#usable_data, valuation_data = train_test_split(filenames,test_size=0.1)

print(time.time()-clock)
clock=time.time()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")

net={"model":[],"optimizer":[]}
for i in range(number_of_models):
    learning_rate= starting_learning_rate*(1+0.25*(i-int(number_of_models/2))) # ex with 4 models starting_learning_rate * [0.5,0.75,1,1.25,1.5]
    print(learning_rate)
    net["model"].append(trackNet(input_size,hidden_size1,hidden_size2and3,output_size).to(device))
    net["optimizer"].append(torch.optim.NAdam(net["model"][i].parameters(), lr=learning_rate))

print(net)
loss_fn = nn.HuberLoss()


clock=time.time()


poiss_obj=TweedieDevianceScore(power=0).to(device)
r2_obj=R2Score().to(device)

all_files=[]
for filename in filenames:
    X,Y = load_data_as_tensor_v2(tracks_dir, racing_line_dir, filename, with_thetas, with_normal_lenght, total_foresight, sampling)
    
    X,Y = X, Y = X.to(device), Y.to(device)
    all_files.append((X,Y))
usable_data, test_data = train_test_split(all_files,test_size=0.2)

#splitting data, insert in loop to change every epoch
#train_data,validation_data = train_test_split(usable_data,test_size=0.1)
n_splits=5

all_indexes=np.arange(len(usable_data))
np.random.shuffle(all_indexes)
split_indexs=np.split(all_indexes,n_splits)

# print(len(split_indexs))#list of kfolds
# print(len(split_indexs[0]))#single kfold with list of indeces aka tracks
# print(len(split_indexs[1]))#single kfold with list of indeces aka tracks
# print(len(usable_data[split_indexs[0][0]]))# single track with both X and Y
# print(len(usable_data[split_indexs[0][0]][0]))# single track all Xs 
# print(len(usable_data[split_indexs[0][0]][1]))# single track all Ys 
# print(len(usable_data[split_indexs[0][0]][0][0]))#single data point


loss_hist=[]
r2_hist=[]
pois_hist=[]

loss_train_hist=[]
r2_train_hist=[]
pois_train_hist=[]

print("time to instantiate model and data: ", time.time()-clock)
old_data=None
#TODO implementa il validation set e 
for t in range(epochs):
    epoch_clock=time.time()
    print(f"Epoch {t+1}\n-------------------------------")

    evaluation_fold=t%n_splits
    train_idx=np.concatenate([split_indexs[j] for j in range(len(split_indexs)) if j != evaluation_fold])
    print(len(split_indexs[0]))
    if old_data!=None:
        print("is something wrong: ", collections.Counter(train_idx)==collections.Counter(old_data))

    train_data=[usable_data[i] for i in train_idx]
    validation_data=[usable_data[i] for i in split_indexs[t%n_splits]]
    
    
    tot_time_predict=train(train_data,net,loss_fn)
    
    best_model_weights, best_loss=evaluation(validation_data,net,loss_fn)
    
    test_clock=time.time() 
    
    avg_loss,r2,poiss=test(test_data,net["model"][0],loss_fn)
    avg_loss_train,r2_train,poiss_train=test(train_data,net["model"][0],loss_fn)

    print(f"Test Error:        Avg loss: {avg_loss:>8f}, r2: {r2:>8f}, mean poisson deviance: {poiss:>8f}")
    print(f"Train Error:       Avg loss: {avg_loss_train:>8f}, r2: {r2_train:>8f}, mean poisson deviance: {poiss_train:>8f}")
    if t!=0:
        print(f"miglioramento del: Avg loss: {(loss_hist[t-1]-avg_loss)/loss_hist[t-1]*100:>8f}% r2: {(r2-r2_hist[t-1])/abs(r2_hist[t-1])*100:>8f}% mean poisson deviance: {(pois_hist[t-1]-poiss)/abs(pois_hist[t-1])*100:>8f}%")
    
    old_data=train_data.copy()
    
    loss_hist.append(avg_loss)
    r2_hist.append(r2)
    pois_hist.append(poiss)

    loss_train_hist.append(avg_loss_train)
    r2_train_hist.append(r2_train)
    pois_train_hist.append(poiss_train)

    epoch_time=time.time()-epoch_clock
    test_time=time.time()-test_clock
    print("time for epoch execution: ", epoch_time, " of which spent testing: ",test_time," spent training: ",tot_time_predict," waiting for various loads in train: ", epoch_time-tot_time_predict-test_time)
    
print("Done!")
print("time to Train: ", time.time()-clock)

#loss_hist = [t.detach().cpu().item() for t in loss_hist]
r2_hist   = [t.detach().cpu().item() for t in r2_hist]
pois_hist = [t.detach().cpu().item() for t in pois_hist]

#loss_train_hist = [t.detach().cpu().item() for t in loss_train_hist]
r2_train_hist   = [t.detach().cpu().item() for t in r2_train_hist]
pois_train_hist = [t.detach().cpu().item() for t in pois_train_hist]

torch.save(net["model"][0].state_dict(), "track_model.pt")
np.savetxt("20_total_foresight_select_best_model_and_lr_stable_no_dist_4_models_lr0,003test.csv", np.column_stack((loss_hist,r2_hist,pois_hist)), fmt="%.6f", delimiter=",")
np.savetxt("20_total_foresight_select_best_model_and_lr_stable_no_dist_4_models_lr0,003train.csv", np.column_stack((loss_train_hist,r2_train_hist,pois_train_hist)), fmt="%.6f", delimiter=",")