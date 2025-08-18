from correction_of_track import correction_of_track
from racetrack_library import load_racing_line_modified,load_track_points
import numpy as np
from os.path import join
import mpmath as mpm
                                                                    #difference between v1 and v2
def racetrack_feature_extraction(tracks_dir,racing_line_dir,filename,with_distances):

    #ottieni i punti dal file
    track_path = join(tracks_dir, filename)
    racing_line_path = join(racing_line_dir, filename)
    points =   mpm.matrix(load_track_points(track_path))
    
    #raceline  è gia rappresentato come [0,1] ovvero una posizione relativa sulla normale rispettiva
    raceline = mpm.matrix(load_racing_line_modified(racing_line_path))

    #"corregge" i punti (in realtà se il tracciato è a posto non fa nulla e thetas=np.zeros)
    left_x, left_y, points, right_x, right_y, thetas, normals=correction_of_track(points)
    
    #ottiene la lunghezza delle normali
    l=mpm.matrix(left_x.rows,1)
    for i in range(len(right_x)):
        l[i]=mpm.sqrt((right_x[i]-left_x[i])**2+(right_y[i]-left_y[i])**2)

    #normal= np.array([-dy, dx])RE

    derivates=mpm.zeros(normals.rows, 2)
    
    derivates[:,0]= normals[:,1]
    derivates[:,1]= normals[:,0]
    derivates[:,1]=-derivates[:,1]
    
    len_derivates=len(normals)
    len_points=len(points)
    distances=mpm.matrix(len_derivates,1)
    #ottiene l'angolo tra 2 normali consecutive (in teoria TODO controllare)
    #!!! radians !!!
    alpha=mpm.matrix(len_derivates,1)
    for i in range(len_derivates-1):
        # dot=mpm.fdot(derivates[i,:],derivates[i+1,:])
        # if(dot<=0):
        #     print(dot)
        # alpha[i]=mpm.acos(dot)
        cross = derivates[i,1]*derivates[i+1,0] - derivates[i,0]*derivates[i+1,1]
        alpha[i]=mpm.asin(cross)

        distances[i]=mpm.sqrt((points[i+1,0]-points[i,0])**2+(points[i+1,1]-points[i,1])**2)

    # dot=mpm.fdot(normals[0,:],normals[len_derivates-1,:])
    # alpha[len(normals)-1]=mpm.acos(dot)
    cross = derivates[len_derivates-1,1]*derivates[0,0] - derivates[len_derivates-1,0]*derivates[0,1]
    alpha[len_derivates-1]=cross
    distances[len(normals)-1]=mpm.sqrt((points[0,0]-points[len_points-1,0])**2+(points[0,1]-points[len_points-1,1])**2)

    s=mpm.mpf("0")
    for i in range(len(alpha)):
        s+=alpha[i]
    

    #riconverto in np perchè piu semplice poi passare a tensor
    l=np.array(l.tolist(), dtype=np.float64)
    alpha=np.array(alpha.tolist(), dtype=np.float64)
    thetas=np.array(thetas.tolist(), dtype=np.float64)
    distances=np.array(distances.tolist(), dtype=np.float64)
    raceline=np.array(raceline.tolist(), dtype=np.float64)

    if with_distances!=0:
        return l,alpha,thetas,distances,raceline
    return l,alpha,thetas,raceline

#TEST
if __name__ == "__main__":
    tracks_dir = "tracks/train/tracks"
    racing_line_dir = "tracks/train/racelinesCorrected"
    filename="Shanghai.csv"    
    
    l,alpha,thetas,distances,raceline=racetrack_feature_extraction(tracks_dir,racing_line_dir,filename,1)

    print(len(l))
    print(len(alpha))
    print(len(thetas))
    print(len(raceline))