from correction_of_track import correction_of_track
from racetrack_library import load_racing_line_modified,load_track_points
import numpy as np
from os.path import join

def feature_extract(tracks_dir,racing_line_dir,filename):

    track_path = join(tracks_dir, filename)
    racing_line_path = join(racing_line_dir, filename)
    points =   np.array(load_track_points(track_path))
    raceline = np.array(load_racing_line_modified(racing_line_path))

    left_x, left_y, points, right_x, right_y, thetas, normals=correction_of_track(points)

    l=np.sqrt(np.power(right_x-left_x,2) + np.power(right_y-left_y,2)) 

    #normal= np.array([-dy, dx])
    derivates=np.zeros(normals.shape)
    
    derivates[:,[0,1]]= normals[:,[1,0]]
    derivates[:,1]=-derivates[:,1]
    
    #!!! radians !!!
    alpha=[]
    for i in range(len(derivates)-1):
        alpha.append(np.arccos(np.dot(derivates[i],derivates[i+1])))
    alpha.append(np.arccos(np.dot(derivates[-1],derivates[0])))

    return l,alpha,thetas,raceline

if __name__ == "__main__":
    tracks_dir = "tracks/train/tracks"
    racing_line_dir = "tracks/train/racelinesCorrected"
    filename="NorisringFlippedX.csv"    
    
    l,alpha,thetas,raceline=feature_extract(tracks_dir,racing_line_dir,filename)

    print(len(l))
    print(len(alpha))
    print(len(thetas))
    print(len(raceline))