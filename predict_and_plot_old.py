#plotta i predict
#ricalcola il modello per 3 
#modifica 
#modifica un po il predict in modo che riceva solo la posizione iniziale e riutilizzi le sue predizioni per calcolare gruppo di dati successivo

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
from racetrack_library import *
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


#track_model.pt = total_foresight=foreward_foresight=20 total_sampling=foreward_sampling=4
#track_model_very_small.pt =  total_foresight=foreward_foresight=3 total_sampling=foreward_sampling=2
#track_model_using_only_predict.pt = total_foresight=foreward_foresight=20 total_sampling=foreward_sampling=4
symmetric=0
also_current_position=0
total_foresight=3 #basically f=total_foresight/2 NOTE must be even number if symmetric
foreward_foresight=3
total_sampling=2 #basically total_sampling
foreward_sampling=2

with_thetas=0 #yes=1 no=0
with_normal_dists=0 #difference between v1 and v2
input_size=(2+with_normal_dists+with_thetas)*(total_foresight+1)
hidden_size1=450
hidden_size2and3=200
#if you want it to be symmetric either put symmetric+1 or let it be (total_sampling)/2
#if you want it to be only foreward (no current position) let it be sampling+1
output_size=total_sampling+(1*also_current_position)


tracks_dir = "tracks/train/featureExtracted"
racing_line_dir = "tracks/train/racelinesCorrected"
#uso il singolo circuito come batch? si dai
filenames =np.array( [f for f in os.listdir(tracks_dir) if os.path.isfile(os.path.join(tracks_dir, f))])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")

# net={"model":[],"optimizer":[]}
# for i in range(number_of_models):
#     learning_rate= starting_learning_rate*(1+0.25*(i-int(number_of_models/2))) # ex with 4 models starting_learning_rate * [0.5,0.75,1,1.25,1.5]
#     print(learning_rate)
#     net["model"].append(TrackNetConditioned(input_size,hidden_size1,hidden_size2and3,output_size).to(device))
#     net["optimizer"].append(torch.optim.NAdam(net["model"][i].parameters(), lr=learning_rate))
# print(net)
loss_fn = nn.HuberLoss()

model = TrackNetConditioned(input_size, hidden_size1, hidden_size2and3, output_size).to(device)
model.load_state_dict(torch.load("small_track_model_using_only_predict_slower_transformation_of_data.pt", map_location=device))

poiss_obj=TweedieDevianceScore(power=0).to(device)
r2_obj=R2Score().to(device)
all_files=[]
print("asymmetric data")
for filename in filenames:                                                                 #with normal lengths
    X,Y = load_data_as_tensor_asymmetric(tracks_dir, racing_line_dir, filename, with_thetas, 1, total_foresight, foreward_foresight, total_sampling, foreward_sampling)
    # X shape: [lentrack, inputsize/2, 3]
    normal_dist_list = X[..., 2:]   # [lentrack, inputsize/2, 1]
    normal_dist_list = normal_dist_list[:, 0]
    X = X[..., :2]    # [lentrack, inputsize/2, 2]
    X,Y = X, Y = X.to(device), Y.to(device)
    current_positions=Y[:,total_sampling-foreward_sampling].detach().clone()
    Y=Y[:,(1-also_current_position):]
    all_files.append((X,Y,current_positions))
usable_data, test_data = train_test_split(all_files,test_size=0.2)

model.eval()
total_loss = 0.0
total_r2 = 0.0
total_poiss = 0.0
with torch.no_grad():
    for file in all_files:
        # carica i dati e li porta sul device corretto
        X,Y,current_positions = file
        current_position=current_positions[0]
        pred = torch.zeros_like(Y)
        save_pred_for_plot = torch.zeros_like(current_positions)
        for i in range(len(X)):
        # forward
            
            pred_single = model(X[i].flatten(), current_position)    # appiattisce input se necessario  #change between current_positions[i] and current_position 
            pred[i] = pred_single
            current_position = pred_single[0].detach().clone()
            save_pred_for_plot[i]=pred_single[0].detach().clone()
        # calcola loss
        loss = loss_fn(pred, Y)
        print(loss)
        total_loss += loss.item()
        r2_obj.update(pred, Y)
        total_r2+=r2_obj.compute()
        r2_obj.reset()
        total_poiss+=poiss_obj(pred,Y)
        poiss_obj.reset()
        fig, axs = plt.subplots(1, 1, figsize=(6,12))

        # Ensure normal_lengths_list is a numpy array
        normal_dist_list = normal_dist_list.cpu().numpy() if hasattr(normal_dist_list, device) else np.array(normal_dist_list)
        lengths = X[:, 0, 0].cpu().numpy() if hasattr(X, device) else np.array(X[:, 0, 0])
        alphas = X[:, 0, 1].cpu().numpy() if hasattr(X, device) else np.array(X[:, 0, 1])
        current_positions = current_positions.cpu().numpy() if hasattr(current_positions, device) else np.array(current_positions)
        save_pred_for_plot=save_pred_for_plot.cpu().numpy() if hasattr(save_pred_for_plot, device) else np.array(save_pred_for_plot)

        plot_tracks_from_features(axs,lengths,alphas,normal_dist_list,current_positions,save_pred_for_plot,"boh")
        plt.show()
    


            

# media sulla lunghezza del test set
avg_loss = total_loss / len(all_files)
avg_r2=total_r2/len(all_files)
avg_poiss=total_poiss/len(all_files)

print(avg_loss,avg_r2,avg_poiss)