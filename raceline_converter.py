#objective: the NN needs for the racing line to be relative to each Normal of the circuit
#HOW:
#creo una bezier curve tra 2 punti e l'incrocio delle loro tangenti.
#trovo dove questa curva interseca la normale vicina
#TOSOLVE: come faccio a sapere che cosa è tra un punto della raceline e l'altra?
from racetrack_library import curva_direzione,load_track_points,load_racing_line_points, plot_tracks,intersect, calc_intersect, add_points
from correction_of_track import correction_of_track
import matplotlib.pyplot as plt
import os
import numpy as np


def add_extra_points_with_derivatives():
    new_raceline=raceline.copy()
    added=0
    new_raceline=[]
    for i in range(1,len(raceline)-2):
        curve=curva_direzione(raceline[i-1],raceline[i],raceline[i+1])
        new_raceline.append(raceline[i])
        if curve!=0:
           x,y=add_points(raceline[i-1],raceline[i],raceline[i+1],raceline[i+2])
           new_raceline.append((x,y))

    new_raceline=np.array(new_raceline)
    # print(new_raceline)
    return new_raceline
 
def raceline_on_normals(points,raceline):
    
    left_x, left_y, points, right_x, right_y, thetas, normals=correction_of_track(points)
    
    #right_x, right_y, left_x, left_y, center_x, center_y, normals=metodo2(points)
    

    left_x_copy=left_x.copy()
    left_y_copy=left_y.copy()
    
    right_x_copy=right_x.copy()
    right_y_copy=right_y.copy()
    copy_raceline=raceline.copy()

    intersec_points=[]


    #CASI LIMITI prima e ultima linea
    #prima


    boolInters=intersect([left_x_copy[0],left_y_copy[0]],[right_x_copy[0],right_y_copy[0]],copy_raceline[-1],copy_raceline[0])

    if boolInters:
        xi,yi,t=calc_intersect([left_x_copy[0],left_y_copy[0]],[right_x_copy[0],right_y_copy[0]],copy_raceline[-1],copy_raceline[0])

        intersec_points.append(t)

        left_x_copy=left_x_copy[1:]
        left_y_copy=left_y_copy[1:]

        right_x_copy=right_x_copy[1:]
        right_y_copy=right_y_copy[1:]

    #ultima
    boolInters=intersect([left_x_copy[-1],left_y_copy[-1]],[right_x_copy[-1],right_y_copy[-1]],copy_raceline[-1],copy_raceline[0])

    if boolInters:
        xi,yi,t=calc_intersect([left_x_copy[-1],left_y_copy[-1]],[right_x_copy[-1],right_y_copy[-1]],copy_raceline[-1],copy_raceline[0])

        intersec_points.append(t)

        left_x_copy=left_x_copy[:-1]
        left_y_copy=left_y_copy[:-1]

        right_x_copy=right_x_copy[:-1]
        right_y_copy=right_y_copy[:-1]

    while len(copy_raceline) != 1 and len(left_x_copy) !=0:
        boolInters=intersect([left_x_copy[0],left_y_copy[0]],[right_x_copy[0],right_y_copy[0]],copy_raceline[0],copy_raceline[1])
        # print(boolInters)
        # print("raceline: ",copy_raceline[0],copy_raceline[1],\
        #       "\n normale: ",[left_x_copy[0],left_y_copy[0]],[right_x_copy[0],right_y_copy[0]] )
        #print(len(copy_raceline), len(left_x_copy))
        xi,yi,t=calc_intersect([left_x_copy[0],left_y_copy[0]],[right_x_copy[0],right_y_copy[0]],copy_raceline[0],copy_raceline[1])
        if boolInters:
            #portion of the normal it is in in [0,1]
            intersec_points.append(t)

            left_x_copy=left_x_copy[1:]
            left_y_copy=left_y_copy[1:]

            right_x_copy=right_x_copy[1:]
            right_y_copy=right_y_copy[1:]

            #print("calcolato: ",xi,yi)
        else:
            copy_raceline=copy_raceline[1:]
            #print(copy_raceline[0])
    #print(intersec_points)
    intersec_points=np.array(intersec_points)

    return left_x, left_y, points, right_x, right_y, thetas, normals, intersec_points

if __name__ == "__main__":
    tracks_dir = "tracks/train/tracks"
    #filename = "Norisring.csv"
    racing_line_dir = "tracks/train/racelines"
    corrected_racing_line_dir="tracks/train/racelinesCorrected"
    correctones=0
    filenames = [f for f in os.listdir(tracks_dir) if os.path.isfile(os.path.join(tracks_dir, f))]
    for filename in filenames:
        track_path = os.path.join(tracks_dir, filename)
        racing_line_path= os.path.join(racing_line_dir, filename)
        points = np.array(load_track_points(track_path))
        raceline=np.array(load_racing_line_points(racing_line_path))

        #print(filename)
        left_x, left_y, points, right_x, right_y, thetas, normals, intersec_points=raceline_on_normals(points,raceline)
        raceline_path = os.path.join(corrected_racing_line_dir,filename)
        np.savetxt(raceline_path, intersec_points, fmt="%.6f", delimiter=",")
        #print(intersec_points)
        
        # if(len(intersec_points)!=len(left_x)):
        #     print(filename, len(intersec_points))
        #     fig, ax = plt.subplots(1, 1, figsize=(6,12))
        #     plot_tracks(ax,left_x,left_y,points,right_x,right_y,intersec_points,filename)
        #     plt.show()
        # else:
        #     correctones+=1
    #print(correctones)
    
# THE DATASETS HAD PROBLEMS IN points
# Catalunya.csv 838+1
# YasMarina.csv 555
# Norisring.csv 330
# Austin.csv 134
# Shanghai.csv 1090
# YasMarina.csv 567 
# YasMarina.csv 760 
# YasMarina.csv 949 
# YasMarina.csv 974
# Norisring.csv 331
# Austin.csv 518
# Austin.csv 519
