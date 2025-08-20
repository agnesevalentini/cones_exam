from racetrack_feature_extraction_v2 import racetrack_feature_extraction
from racetrack_feature_pre_extracted import racetrack_feature_pre_extracted
import numpy as np
import torch
import time
def load_data_as_tensor(tracks_dir,racing_line_dir,filename,with_thetas,with_dists,total_foresight,sampling):
    if tracks_dir=="tracks/train/featureExtracted":
        res=racetrack_feature_pre_extracted(tracks_dir,racing_line_dir,filename,with_dists) 
    else:
        res=racetrack_feature_extraction(tracks_dir,racing_line_dir,filename,with_dists)
    if(with_dists!=0):
        l,alpha,thetas,dists,raceline=res
    else:
        l,alpha,thetas,raceline=res
    # depends if we know wehter there are modified normals in the tracks
    if with_dists==1 and with_thetas==1:
        features = np.array((l, alpha,dists,thetas))
    elif with_dists==1 and with_thetas==0:
        features = np.array((l,alpha,dists))
    elif with_dists==0 and with_thetas==1:
        features = np.array((l, alpha,thetas))
    else: 
        features = np.array((l, alpha))
    #print(features.shape)

    track_length=len(l)
    # i need to know how many track points do i need to know before and after the normal we are focussed on
    half_in = total_foresight //2
    half_out = sampling

    #crea una lista di indici di tutt i punti
    centers = np.arange(track_length)

    #crea gli indici per ogni training sample, ovvero "puntocentrale" e indici precedenti e successivi:
    # indici input: (track_length, input_size)
    input_idx = (centers[:, None] + np.arange(-half_in, half_in+1)) % track_length
    #print("input_idx",input_idx.shape)

    #come sopra
    # indici output: (track_length, output_size)
    output_idx = (centers[:, None] + np.arange(-half_out, half_out+1)) % track_length
    #print("output_idx",output_idx.shape)
    
    # costruisci le feature
    #features = [l, alpha]  # aggiungi thetas se serve
    X = np.stack([np.take(f, input_idx) for f in features], axis=-1)   # shape (track_length, input_size, n_features)
    # costruisci i target
    Y = np.take(raceline, output_idx)    # shape (track_length, output_size)
    # converti in tensori

    X = torch.tensor(X, dtype=torch.float64)
    Y = torch.tensor(Y, dtype=torch.float64)
    
    return X,Y

if __name__== "__main__":
    t=time.time()
    #tracks_dir = "tracks/train/tracks"
    tracks_dir = "tracks/train/featureExtracted"
    racing_line_dir = "tracks/train/racelinesCorrected"
    filename="Shanghai.csv"    
    
    X,Y =load_data_as_tensor(tracks_dir,racing_line_dir,filename,0,1,70,4)

    print(X.shape,Y.shape)
    print(Y)
    print(time.time()-t)