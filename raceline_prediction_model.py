import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import numpy as np
from load_data_as_tensor import load_data_as_tensor
import time
total_foresight=70 #basically f=35
s=4
with_thetas=0 #yes=1 no=0
input_size=(2+with_thetas)*(total_foresight+1)
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
        x = self.flatten(x)
        logits = logits = self.model_stack(x)
        return logits

def train(filenames, model, loss_fn, optimizer):
    #il singolo filename è una batch di training data
    #se faccio questo vuol dire che in pratica lo stesso dato deve essere utilizzato come lable su neuroni diversi
    
    model.train()
    
    for batch, filename in enumerate(filenames):
        #print(batch)
        X,Y = load_data_as_tensor(tracks_dir,racing_line_dir,filename,with_thetas,total_foresight,sampling)
        track_length=X.shape[0]
        #print(X.shape, Y.shape)
        
        batch_X = X.view(track_length, -1)   # shape: (track_length, input_size * n_features)
        pred = model(batch_X)                # shape: (track_length, output_size)
        loss = loss_fn(pred, Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(filenames):>5d}]")
    
    return    

def test(filenames, model, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for filename in filenames:
            # carica i dati e li porta sul device corretto
            X, Y = load_data_as_tensor(tracks_dir, racing_line_dir, filename, with_thetas, total_foresight, sampling)
            X, Y = X.to(device), Y.to(device)

            # forward
            pred = model(X.view(X.shape[0], -1))  # appiattisce input se necessario

            # calcola loss
            loss = loss_fn(pred, Y)
            total_loss += loss.item()

    # media sulla lunghezza del test set
    avg_loss = total_loss / len(filenames)

    print(f"Test Error: Avg loss: {avg_loss:>8f}")

    return avg_loss

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
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data,net,loss_fn,optimizer)
    test(test_data,net,loss_fn)
print("Done!")
print("time to Train: ", time.time()-clock)

torch.save(net.state_dict(), "track_model.pt")
