import torch
import torch.nn as nn
from torchmetrics import TweedieDevianceScore
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import numpy as np
from load_data_as_tensor import load_data_as_tensor
from load_data_as_tensor_v2 import load_data_as_tensor as load_data_as_tensor_v2
import time
total_foresight=70 #basically f=35
s=4
with_thetas=0 #yes=1 no=0
with_normal_lenght=1 #difference between v1 and v2
input_size=(2+with_normal_lenght+with_thetas)*(total_foresight+1)
hidden_size1=450
hidden_size2and3=200
sampling=4
output_size=2*sampling+1
learning_rate = 0.003 # learning rate

class trackNet(nn.Module):
    def __init__(self,input_size, hidden_size1,hidden_size2and3, output_size):
        
        super().__init__()
        self.flatten = nn.Flatten()
        self.model_stack=nn.Sequential(
        nn.Linear(input_size,hidden_size1),
        nn.Sigmoid(),
        nn.Linear(hidden_size1,hidden_size2and3),
        nn.Sigmoid(),
        nn.Linear(hidden_size2and3,hidden_size2and3),
        nn.Sigmoid(),
        nn.Linear(hidden_size2and3,output_size),
        nn.Hardsigmoid()
        )   

    def forward(self, x):
        # x = self.flatten(x)
        logits = logits = self.model_stack(x)
        return logits

def train(filenames, model, loss_fn, optimizer):
    #il singolo filename è una batch di training data
    #se faccio questo vuol dire che in pratica lo stesso dato deve essere utilizzato come lable su neuroni diversi
    
    model.train()
    
    for batch, filename in enumerate(filenames):
        #print(batch)
        if with_normal_lenght==0:
            X,Y = load_data_as_tensor(tracks_dir,racing_line_dir,filename,with_thetas,total_foresight,sampling)
        else:
            X,Y = load_data_as_tensor_v2(tracks_dir,racing_line_dir,filename,with_thetas,total_foresight,sampling)
        track_length=X.shape[0]
        #print(X.shape, Y.shape)
        
        batch_X = X.view(track_length, -1)   # shape: (track_length, input_size * n_features)
        pred = model(batch_X)                # shape: (track_length, output_size)
        loss = loss_fn(pred, Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 600 == 0:
            loss, current = loss.item(), (batch + 1)
            r2=r2_score(Y.detach().numpy(),pred.detach().numpy())
            print(f"loss: {loss:>7f}, r2: {r2}  [{current:>5d}/{len(filenames):>5d}]")
    
    return    

def test(filenames, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_r2 = 0.0
    total_poiss = 0.0
    with torch.no_grad():
        for filename in filenames:
            # carica i dati e li porta sul device corretto
            if with_normal_lenght==0:
                X,Y = load_data_as_tensor(tracks_dir,racing_line_dir,filename,with_thetas,total_foresight,sampling)
            else:
                X,Y = load_data_as_tensor_v2(tracks_dir,racing_line_dir,filename,with_thetas,total_foresight,sampling)
            X, Y = X.to(device), Y.to(device)

            # forward
            pred = model(X.view(X.shape[0], -1))  # appiattisce input se necessario

            # calcola loss
            loss = loss_fn(pred, Y)
            total_loss += loss.item()
            total_r2+=r2_score(Y,pred)
            total_poiss+=poiss_obj(pred,Y)
            

    # media sulla lunghezza del test set
    avg_loss = total_loss / len(filenames)
    avg_r2=total_r2/len(filenames)
    avg_poiss=total_poiss/len(filenames)
    return avg_loss,avg_r2,avg_poiss

clock=time.time()
tracks_dir = "tracks/train/tracks"
racing_line_dir = "tracks/train/racelinesCorrected"
#uso il singolo circuito come batch? si dai
filenames =np.array( [f for f in os.listdir(tracks_dir) if os.path.isfile(os.path.join(tracks_dir, f))])
usable_data, valuation_data = train_test_split(filenames,test_size=0.1)
train_data,test_data=train_test_split(usable_data,test_size=0.2)

print(time.time()-clock)
clock=time.time()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

net = trackNet(input_size,hidden_size1,hidden_size2and3,output_size).to(device)
print(net)
loss_fn = nn.HuberLoss()
optimizer = torch.optim.NAdam(net.parameters(), lr=learning_rate)

print("time to instantiate model: ", time.time()-clock)
clock=time.time()
epochs = 100

poiss_obj=TweedieDevianceScore(power=1)
for t in range(epochs):
    epoch_clock=time.time()
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data,net,loss_fn,optimizer)
    
    avg_loss,r2,poiss=test(test_data,net,loss_fn)
    print(f"Test Error: Avg loss: {avg_loss:>8f}, r2: {r2:>8f}, mean poisson deviance: {poiss:>8f}")
    if t!=0:
        print(f"miglioramento del: Avg loss: {(prec-avg_loss)/prec*100:>8f}% r2: {(prec_r2-r2)/prec_r2*100:>8f}% mean poisson deviance: {(prec_poiss-poiss)/1*100:>8f}%")
    prec=avg_loss
    prec_r2=r2
    prec_poiss=poiss
    print("time for epoch execution: ", time.time()-epoch_clock)
    
print("Done!")
print("time to Train: ", time.time()-clock)

torch.save(net.state_dict(), "track_model.pt")
