import os
import matplotlib.pyplot as plt
import numpy as np
#from trajectory_planning_helpers.calc_normal_vectors import calc_normal_vectors
from trajectory_planning_helpers.check_normals_crossing import check_normals_crossing
from test_show_track import metodo2, load_track_points,load_racing_line_points, plot_tracks, correttore, curva_direzione


tracks_dir = "tracks/train/tracks"
filenames = [f for f in os.listdir(tracks_dir) if os.path.isfile(os.path.join(tracks_dir, f))]


racing_line_dir = "tracks/train/racelines"


for filename in filenames:
    #filename="ShanghaiFlippedX.csv"
# DATA CORRECTION

    track_path = os.path.join(tracks_dir, filename)
    racing_line_path= os.path.join(racing_line_dir, filename)

    points = np.array(load_track_points(track_path))
    raceline=np.array(load_racing_line_points(racing_line_path))


    w_right_x, w_right_y, w_left_x, w_left_y, center_x, center_y, normals = metodo2(points)
    thetas=np.zeros(len(points))


    p=check_normals_crossing(np.array(points), np.array(normals), 6,True)

    if(p!=[]):
        fig, axs = plt.subplots(1, 2, figsize=(6,12))
        problems=[p[0]]
        for i in range(1, len(p)):
            if(p[i]!=p[i-1]):
                problems.append(p[i])

        direzione = curva_direzione(points[problems[0]-1], points[problems[0]], points[problems[0]+1])

        new_points=points.copy()
        if(direzione==-1):
            w_left_x_mod, w_left_y_mod, w_right_x_mod, w_right_y_mod, t, new_normals = correttore(problems, w_left_x, w_left_y, points, w_right_x, w_right_y)
            for i in range(len(problems)):
                thetas[problems[i]]=t[i]
                new_points[problems[i]][3 ]=np.sqrt(np.power(w_left_x_mod[problems[i]]-points[problems[i]][0],2) + np.power(w_left_y_mod[problems[i]]-points[problems[i]][1],2)) 
                #TODO decide if also save the new normals
        elif(direzione==1):
            w_right_x_mod, w_right_y_mod, w_left_x_mod, w_left_y_mod, t, new_normals = correttore(problems, w_right_x, w_right_y, points, w_left_x, w_left_y)
            for i in range(len(problems)):
                thetas[problems[i]]=t[i]
                new_points[problems[i]][2]=np.sqrt(np.power(w_left_x_mod[problems[i]]-points[problems[i]][0],2) + np.power(w_left_y_mod[problems[i]]-points[problems[i]][1],2)) 
                #TODO decide if also save the new normals

        plot_tracks(axs[0],w_left_x,w_left_y,points,w_right_x,w_right_y,raceline,filename)
        plot_tracks(axs[1],w_left_x_mod,w_left_y_mod,points,w_right_x_mod,w_right_y_mod,raceline,filename)
        plt.suptitle(f"Track: {filename}")
        plt.show()

    # scaling, flipping, and reversing