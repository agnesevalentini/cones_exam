import os
import matplotlib.pyplot as plt
#import numpy as np
import mpmath as mpm
#from trajectory_planning_helpers.calc_normal_vectors import calc_normal_vectors
from racetrack_library import metodo2, load_track_points,load_racing_line_points, plot_tracks, correttore, curva_direzione,intersect

def correction_of_track(points):
    w_right_x, w_right_y, w_left_x, w_left_y, center_x, center_y, normals = metodo2(points)
    thetas=mpm.matrix([0]*len(points))

    #controllo se le normali si intreccianos
    p=[]
    for i in range(1,len(w_right_x)-1):
        if intersect([w_right_x[i-1],w_right_y[i-1]],[w_left_x[i-1],w_left_y[i-1]],[w_right_x[i],w_right_y[i]],[w_left_x[i],w_left_y[i]]) or \
           intersect([w_right_x[i],w_right_y[i]],[w_left_x[i],w_left_y[i]],[w_right_x[i+1],w_right_y[i+1]],[w_left_x[i+1],w_left_y[i+1]]):
            p.append(i)
    
    if(p!=[]):
        
        problems=[p[0]]
        for i in range(1, len(p)):
            if(p[i]!=p[i-1]):
                problems.append(p[i])

        direzione = curva_direzione(points[problems[0]-1], points[problems[0]], points[problems[0]+1])

        new_points=points.copy()
        new_normals=normals.copy()
        #print(direzione)
        if(direzione==-1):
            w_left_x_mod, w_left_y_mod, w_right_x_mod, w_right_y_mod, t, nn = correttore(problems, w_left_x, w_left_y, points, w_right_x, w_right_y)
            for i in range(len(problems)):
                if mpm.isnan(t):
                    print(filename)
                thetas[problems[i]]=t[i]
                new_points[problems[i],3]=mpm.sqrt(mpm.power(w_left_x_mod[problems[i]]-points[problems[i],0],2) + mpm.power(w_left_y_mod[problems[i]]-points[problems[i],1],2)) 
                x,y=nn[i]
                new_normals[problems[i]]=[-x,-y]
        elif(direzione==1):
            w_right_x_mod, w_right_y_mod, w_left_x_mod, w_left_y_mod, t, nn = correttore(problems, w_right_x, w_right_y, points, w_left_x, w_left_y)
            for i in range(len(problems)):
                if mpm.isnan(t):
                    print(filename)
                thetas[problems[i]]=t[i]
                new_points[problems[i],2]=mpm.sqrt(mpm.power(w_left_x_mod[problems[i]]-points[problems[i],0],2) + mpm.power(w_left_y_mod[problems[i]]-points[problems[i],1],2)) 
                x,y=nn[i]
                new_normals[problems[i]]=[x,y]
    
        return  w_left_x_mod, w_left_y_mod, new_points, w_right_x_mod, w_right_y_mod, thetas, new_normals
    return      w_left_x, w_left_y, points, w_right_x, w_right_y,thetas, normals

if __name__ == "__main__":

    
    tracks_dir = "tracks/train/tracks"
    #filenames = [f for f in os.listdir(tracks_dir) if os.path.isfile(os.path.join(tracks_dir, f))]


    racing_line_dir = "tracks/train/racelines"
    filename="NorisringReversed.csv"
    
    track_path = os.path.join(tracks_dir, filename)
    racing_line_path= os.path.join(racing_line_dir, filename)

    points = mpm.matrix(load_track_points(track_path))
    raceline=mpm.matrix(load_racing_line_points(racing_line_path))

    fig, axs = plt.subplots(1, 2, figsize=(6,12))
    w_right_x, w_right_y, w_left_x, w_left_y, center_x, center_y, normals = metodo2(points)
    w_left_x_mod, w_left_y_mod, new_points, w_right_x_mod, w_right_y_mod, thetas, new_normals=correction_of_track(points)    

    #set_diff = np.setdiff1d(new_normals, normals)
    #print(set_diff)

    plot_tracks(axs[0],w_left_x,w_left_y,points,w_right_x,w_right_y,raceline,filename)
    plot_tracks(axs[1],w_left_x_mod,w_left_y_mod,points,w_right_x_mod,w_right_y_mod,raceline,filename)
    plt.suptitle(f"Track: {filename}")
    plt.show()



