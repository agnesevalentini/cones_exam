from correction_of_track import correction_of_track
from racetrack_library import load_racing_line_modified,load_track_points
import numpy as np
from os.path import join


def racetrack_feature_extraction(tracks_dir,racing_line_dir,filename):

    #ottieni i punti dal file
    track_path = join(tracks_dir, filename)
    racing_line_path = join(racing_line_dir, filename)
    points =   np.array(load_track_points(track_path))
    
    #raceline  è gia rappresentato come [0,1] ovvero una posizione relativa sulla normale rispettiva
    raceline = np.array(load_racing_line_modified(racing_line_path))

    #"corregge" i punti (in realtà se il tracciato è a posto non fa nulla e thetas=np.zeros)
    left_x, left_y, points, right_x, right_y, thetas, normals=correction_of_track(points)

    #ottiene la lunghezza delle normali
    l=np.sqrt(np.power(right_x-left_x,2) + np.power(right_y-left_y,2)) 

    #normal= np.array([-dy, dx])
    derivates=np.zeros(normals.shape)
    
    derivates[:,[0,1]]= normals[:,[1,0]]
    derivates[:,1]=-derivates[:,1]
    
    #ottiene l'angolo tra 2 normali consecutive (in teoria TODO controllare)
    #!!! radians !!!
    alpha=[]
    for i in range(len(derivates)-1):
        dot=np.dot(derivates[i],derivates[i+1])
        alpha.append(np.arccos(np.clip(dot, -1.0, 1.0)))
    dot=np.dot(derivates[-1],derivates[0])
    alpha.append(np.arccos(np.clip(dot, -1.0, 1.0)))

    return l,alpha,thetas,raceline


#TEST
if __name__ == "__main__":
    tracks_dir = "tracks/train/tracks"
    racing_line_dir = "tracks/train/racelinesCorrected"
    filename="NorisringFlippedX.csv"    
    
    l,alpha,thetas,raceline=racetrack_feature_extraction(tracks_dir,racing_line_dir,filename)

    print(len(l))
    print(len(alpha))
    print(len(thetas))
    print(len(raceline))