import torch
import torch.nn as nn
from torchmetrics import TweedieDevianceScore, R2Score
from sklearn.model_selection import train_test_split
import copy
import os
import numpy as np
import collections
from load_data_as_tensor_v2 import load_data_as_tensor as load_data_as_tensor_v2, load_data_as_tensor_asymmetric
import time
symmetric=0
also_current_position=0


total_foresight=3 #basically f=total_foresight/2 NOTE must be even number if symmetric
foreward_foresight=3
total_sampling=2 #basically total_sampling
foreward_sampling=2

with_thetas=0 #yes=1 no=0
with_normal_dist=0 #difference between v1 and v2
input_size=(2+with_normal_dist+with_thetas)*(total_foresight+1)
hidden_size1=450
hidden_size2and3=200
#if you want it to be symmetric either put symmetric+1 or let it be (total_sampling)/2
#if you want it to be only foreward (no current position) let it be sampling+1
output_size=total_sampling+(1*also_current_position)
starting_learning_rate = 0.003 # learning rate
epochs = 50
number_of_models=4


if(foreward_foresight>total_foresight or foreward_sampling>total_sampling):
    raise("ERROR you can't have more view foreward then the total")
elif(foreward_sampling!=total_sampling and also_current_position==0):
    raise("ERROR you can't have only foreward sampling and no inplace if the toal sampling is not equal to foreward sampling")
elif(symmetric==1 and also_current_position==0):
    raise("ERROR you can't have only foreward sampling and no inplace if the foresight is symmetric")
class TrackNetConditioned(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2and3, output_size):

        super().__init__()
        
        # main path
        self.fc1 = nn.Linear(input_size, hidden_size1, dtype=torch.float64)
        self.act1 = nn.Sigmoid()
        
        # conditioning branch (takes scalar s -> gamma, beta)
        self.film = nn.Sequential(
            nn.Linear(1, 32, dtype=torch.float64), nn.Sigmoid(),
            nn.Linear(32, 2*hidden_size1, dtype=torch.float64)
        )
        
        # rest of the model
        self.fc2 = nn.Linear(hidden_size1, hidden_size2and3, dtype=torch.float64)
        self.act2 = nn.Sigmoid()
        
        self.fc3 = nn.Linear(hidden_size2and3, hidden_size2and3, dtype=torch.float64)
        self.act3 = nn.Sigmoid()
        
        self.fc4 = nn.Linear(hidden_size2and3, output_size, dtype=torch.float64)
        self.out = nn.Hardsigmoid()
    
    def forward(self, x, s):
        # main path first layer
        h = self.act1(self.fc1(x))   # [B, hidden_size1]
        
        # conditioning
        gb = self.film(s.unsqueeze(-1))  # [B, 2*hidden_size1]
        gamma, beta = torch.chunk(gb, 2, dim=-1)  # [B, hidden_size1] ciascuno
        h = gamma * h + beta
        
        # rest of network
        h = self.act2(self.fc2(h))
        h = self.act3(self.fc3(h))
        logits = self.out(self.fc4(h))
        return logits

def markovian_predictions(model,X,current_positions,Y):
    current_position=current_positions[0]
    pred = torch.zeros_like(Y)
    save_pred_for_plot = torch.zeros_like(current_positions)
    for i in range(len(X)):
    # forward
            
        pred_single = model(X[i].flatten(), current_position)    # appiattisce input se necessario  #change between current_positions[i] and current_position 
        pred[i] = pred_single
        current_position = pred_single[0].detach().clone()
        save_pred_for_plot[i]=pred_single[0].detach().clone()

def train(files, models, loss_fn):
    #il singolo filename è una batch di training data
    #se faccio questo vuol dire che in pratica lo stesso dato deve essere utilizzato come lable su neuroni diversi
    tot_time=0

    best_pred_list=[]
    for batch, file in enumerate(files):
        best_loss=np.inf
        best_pred=None
        for i in range(len(models["model"])):
            model=models["model"][i]
            model.train()

            optimizer=models["optimizer"][i]
            model.train()
            
            #print(batch)
            X,Y,current_positions = file
            track_length=X.shape[0]
            #print(X.shape, Y.shape)

            t_batch=time.time()
            
            batch_X = X.view(track_length, -1)   # shape: (track_length, input_size * n_features)
            pred = model(batch_X, current_positions)                # shape: (track_length, output_size)
            loss = loss_fn(pred, Y)
            if best_loss>loss:
                best_loss=loss
                best_pred=pred[:,total_sampling-foreward_sampling].detach().clone()

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
                print(f"model {i+1} loss: {loss:>7f}, r2: {r2}  [{current:>5d}/{len(files):>5d}], mean time for exec: {tot_time/(len(models['model'])*(batch)+i+1)}")
        best_pred_list.append(best_pred)
    return tot_time, best_pred_list   

def evaluation(filenames, models, loss_fn):
    min_loss=np.inf
    best_model=None
    best_lr=None
    for i in range(len(models["model"])):
        model=models["model"][i]
        
        lr=models["optimizer"][i].param_groups[0]["lr"]

        avg_loss,_,_,best_pred_list = test(filenames,model,loss_fn)
        #somehow find a way to incorporate other things in the decision
        if avg_loss<=min_loss:
            best_model=copy.deepcopy(model.state_dict())
            min_loss=avg_loss
            best_lr=lr
            print("new best model! ",avg_loss)
    for i in range(len(models["model"])):
                        #change with lr if you want convergin learning rate
        learning_rate= starting_learning_rate*(1+0.25*(i-int(len(models["model"])/2))) # ex with 4 models starting_learning_rate * [0.5,0.75,1,1.25,1.5]
        models["model"][i].load_state_dict(best_model)
        models["optimizer"][i].param_groups[0]["lr"]=learning_rate


    return best_model, min_loss,best_pred_list

def test(files, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_r2 = 0.0
    total_poiss = 0.0
    with torch.no_grad():
        best_pred_list=[]
        for file in files:
            # carica i dati e li porta sul device corretto
            X,Y,current_positions = file

            # forward
            pred = model(X.view(X.shape[0], -1),current_positions)    # appiattisce input se necessario 

            # calcola loss
            loss = loss_fn(pred, Y)
            total_loss += loss.item()
            r2_obj.update(pred, Y)
            total_r2+=r2_obj.compute()
            r2_obj.reset()
            total_poiss+=poiss_obj(pred,Y)
            poiss_obj.reset()

            best_pred_list.append(pred[:,total_sampling-foreward_sampling])

    # media sulla lunghezza del test set
    avg_loss = total_loss / len(files)
    avg_r2=total_r2/len(files)
    avg_poiss=total_poiss/len(files)
    return avg_loss,avg_r2,avg_poiss,best_pred_list

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
    net["model"].append(TrackNetConditioned(input_size,hidden_size1,hidden_size2and3,output_size).to(device))
    net["optimizer"].append(torch.optim.NAdam(net["model"][i].parameters(), lr=learning_rate))

print(net)
loss_fn = nn.HuberLoss()


clock=time.time()


poiss_obj=TweedieDevianceScore(power=0).to(device)
r2_obj=R2Score().to(device)

all_files=[]
if symmetric==1:
    print("symmetric data")
    for filename in filenames:
        X,Y = load_data_as_tensor_v2(tracks_dir, racing_line_dir, filename, with_thetas, with_normal_dist, total_foresight, total_sampling)
        
        X,Y = X, Y = X.to(device), Y.to(device)
        current_positions=Y[:,total_sampling-foreward_sampling].detach().clone()
        all_files.append((X,Y,current_positions))
    usable_data, test_data = train_test_split(all_files,test_size=0.2)
else:
    print("asymmetric data")
    for filename in filenames:
        X,Y = load_data_as_tensor_asymmetric(tracks_dir, racing_line_dir, filename, with_thetas, with_normal_dist, total_foresight, foreward_foresight, total_sampling, foreward_sampling)
        
        X,Y = X, Y = X.to(device), Y.to(device)
        current_positions=Y[:,total_sampling-foreward_sampling].detach().clone()
        Y=Y[:,(1-also_current_position):]
        all_files.append((X,Y,current_positions))
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
#print(len(usable_data[split_indexs[0][0]][1][0]))#single data point


loss_hist=[]
r2_hist=[]
pois_hist=[]

loss_train_hist=[]
r2_train_hist=[]
pois_train_hist=[]

print("time to instantiate model and data: ", time.time()-clock)
old_data=None
overall_best_model=None
overall_best_loss=np.inf
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
    
    
    tot_time_predict, new_current_positions_train=train(train_data,net,loss_fn)
    
    for i,idx in enumerate(train_idx):
        # Usare Teacher Forcing Ratio per la combinazione, riduco ad ogni epoca quanto dell'informazione viene dalle label e quanto dalle precedenti predizioni
        usable_data[idx] = (usable_data[idx][0], usable_data[idx][1], ((1-(min(t/epochs,1))) * usable_data[idx][2] + (min(t/epochs,1)) * new_current_positions_train[i]).detach().clone())
        


    best_model_weights, best_loss,new_current_positions_eval=evaluation(validation_data,net,loss_fn)
    for i,idx in enumerate(split_indexs[t%n_splits]):
        usable_data[idx]= (usable_data[idx][0], usable_data[idx][1], ((1-(min(t/epochs,1))) * usable_data[idx][2] + (min(t/epochs,1)) * new_current_positions_eval[i]).detach().clone())
    
    if overall_best_loss>best_loss:
        overall_best_loss=best_loss
        overall_best_model=best_model_weights
    test_clock=time.time() 
    
    avg_loss,r2,poiss,new_current_positions_test=test(test_data,net["model"][0],loss_fn)
    for idx in range(len(test_data)):
        test_data[idx]= (test_data[idx][0], test_data[idx][1], ((1-(min(t/epochs,1))) * test_data[idx][2] + (min(t/epochs,1)) * new_current_positions_test[idx]).detach().clone())
    avg_loss_train,r2_train,poiss_train,_=test(train_data,net["model"][0],loss_fn)

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

#track_model_using_only_predict.pt = total_foresight=foreward_foresight=20 total_sampling=foreward_sampling=4

torch.save(overall_best_model, "small_track_model_using_only_predict_slower_transformation_of_data.pt")
np.savetxt("3_total_foresight_asymetric_2_select_best_model_and_lr_stable_no_dist_4_models_lr0,003test.csv", np.column_stack((loss_hist,r2_hist,pois_hist)), fmt="%.6f", delimiter=",")
np.savetxt("3_total_foresight_asymetric_2_select_best_model_and_lr_stable_no_dist_4_models_lr0,003train.csv", np.column_stack((loss_train_hist,r2_train_hist,pois_train_hist)), fmt="%.6f", delimiter=",")