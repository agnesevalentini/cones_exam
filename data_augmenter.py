#flippedX,flippedY,flippedXY, reversing rotate 30 and 60
#minimum width 5m

import os
import numpy as np
import matplotlib.pyplot as plt
from test_show_track import metodo2, load_track_points,load_racing_line_points

racing_line_dir = "tracks/train/racelines"
tracks_dir = "tracks/train/tracks"


for action in ["flipX","flipY","reverse","rotate","scale"]:
    filenames = [f for f in os.listdir(tracks_dir) if os.path.isfile(os.path.join(tracks_dir, f))]
    for filename in filenames:

        #filename="NorisringFlippedX.csv"
        track_path = os.path.join(tracks_dir, filename)
        racing_line_path= os.path.join(racing_line_dir, filename)
        points = np.array(load_track_points(track_path))
        raceline=np.array(load_racing_line_points(racing_line_path))

        # Flip points and raceline along the y-axis (invert x coordinate)
        if action=="flipX":
            flipped_points = points.copy()
            flipped_points[:, 0] = -flipped_points[:, 0]
            

            flipped_raceline = raceline.copy()
            flipped_raceline[:, 0] = -flipped_raceline[:, 0]

            temp=flipped_points[:,2].copy()
            flipped_points[:,2]=flipped_points[:,3].copy()
            flipped_points[:,3]=temp

            # Save flipped track points
            filenameNoCSV=filename.removesuffix(".csv")
            flipped_track_path = os.path.join(tracks_dir, f"{filenameNoCSV}FlippedX.csv")
            np.savetxt(flipped_track_path, flipped_points, fmt="%.6f", delimiter=",")

            # Save flipped raceline points
            flipped_raceline_path = os.path.join(racing_line_dir, f"{filenameNoCSV}FlippedX.csv")
            np.savetxt(flipped_raceline_path, flipped_raceline, fmt="%.6f", delimiter=",")
        elif action=="flipY":
            flipped_points = points.copy()
            flipped_points[:, 1] = -flipped_points[:, 1]
            

            flipped_raceline = raceline.copy()
            flipped_raceline[:, 1] = -flipped_raceline[:, 1]

            temp=flipped_points[:,2].copy()
            flipped_points[:,2]=flipped_points[:,3].copy()
            flipped_points[:,3]=temp

            # Save flipped track points
            filenameNoCSV=filename.removesuffix(".csv")
            flipped_track_path = os.path.join(tracks_dir, f"{filenameNoCSV}FlippedY.csv")
            np.savetxt(flipped_track_path, flipped_points, fmt="%.6f", delimiter=",")

            # Save flipped raceline points
            flipped_raceline_path = os.path.join(racing_line_dir, f"{filenameNoCSV}FlippedY.csv")
            np.savetxt(flipped_raceline_path, flipped_raceline, fmt="%.6f", delimiter=",")
        elif action=="reverse":
            reversed_points = points[::-1].copy()
            reversed_raceline = raceline[::-1].copy()

            filenameNoCSV = filename.removesuffix(".csv")
            reversed_track_path = os.path.join(tracks_dir, f"{filenameNoCSV}Reversed.csv")
            np.savetxt(reversed_track_path, reversed_points, fmt="%.6f", delimiter=",")

            reversed_raceline_path = os.path.join(racing_line_dir, f"{filenameNoCSV}Reversed.csv")
            np.savetxt(reversed_raceline_path, reversed_raceline, fmt="%.6f", delimiter=",")
        elif action=="rotate" and "Rotated":
            for rotation_angle in [30,60]: #IMPORTANT STAY IN [0,90]
                theta = np.radians(rotation_angle)  # Define rotation_angle as needed
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                            [np.sin(theta),  np.cos(theta)]])

                rotated_points = points.copy()
                rotated_points[:, :2] = np.dot(rotated_points[:, :2], rotation_matrix.T)

                rotated_raceline = raceline.copy()
                rotated_raceline[:, :2] = np.dot(rotated_raceline[:, :2], rotation_matrix.T)

                filenameNoCSV = filename.removesuffix(".csv")
                rotated_track_path = os.path.join(tracks_dir, f"{filenameNoCSV}Rotated{rotation_angle}.csv")
                np.savetxt(rotated_track_path, rotated_points, fmt="%.6f", delimiter=",")

                rotated_raceline_path = os.path.join(racing_line_dir, f"{filenameNoCSV}Rotated{rotation_angle}.csv")
                np.savetxt(rotated_raceline_path, rotated_raceline, fmt="%.6f", delimiter=",")
        elif action=="scale":
            for mul in [1,1.5]:
                mindist=np.mean([np.mean(points[:,2]),np.mean(points[:,3])])
                scale_factor = (2.5 / mindist)*mul
                #print(scale_factor)
                scaled_points = points.copy()
                scaled_raceline=raceline.copy()
                # scaled_points[:, 2] *= scale_factor
                # scaled_points[:, 3] *= scale_factor

                scaling_matrix = np.array([[scale_factor, 0],
                                        [0, scale_factor]])

                right_x, right_y, left_x, left_y, _, _, _ = metodo2(points)
                
                scaled_points[:, :2] = np.dot(scaled_points[:, :2], scaling_matrix.T)
                scaled_raceline[:, :2] = np.dot(scaled_raceline[:, :2], scaling_matrix.T)
                
                right_coords = np.column_stack((right_x, right_y))
                left_coords = np.column_stack((left_x,left_y))
                
                right_coords=np.dot(right_coords,scaling_matrix.T)
                left_coords=np.dot(left_coords,scaling_matrix.T)

                scaled_points[:, 2]=np.sqrt(np.power(right_coords[:,0]-scaled_points[:,0],2) + np.power(right_coords[:,1]-scaled_points[:,1],2))
                scaled_points[:, 3]=np.sqrt(np.power(left_coords[:,0]-scaled_points[:,0],2) + np.power(left_coords[:,1]-scaled_points[:,1],2))
                
                filenameNoCSV = filename.removesuffix(".csv")
                scaled_track_path = os.path.join(tracks_dir, f"{filenameNoCSV}Scaled{scale_factor}.csv")
                np.savetxt(scaled_track_path, scaled_points, fmt="%.6f", delimiter=",")
                scaled_raceline_path = os.path.join(racing_line_dir, f"{filenameNoCSV}Scaled{scale_factor}.csv")
                np.savetxt(scaled_raceline_path, scaled_raceline, fmt="%.6f", delimiter=",")
    # right_x, right_y, left_x, left_y, center_x, center_y, normals = metodo2(rotated_points)

    # fig, axs = plt.subplots(1, 1)
    # plot_tracks(axs,left_x,left_y,rotated_points,right_x,right_y,rotated_raceline, filename)
    # axs.grid()
    # axs.set_aspect('equal', adjustable='box')
    # right_xb, right_yb, left_xb, left_yb, center_xb, center_yb, normalsb = metodo2(points)
    # fig2, axs2 = plt.subplots(1, 1)
    # plot_tracks(axs2,left_xb, left_yb,points,right_xb, right_yb,raceline, filename)
    # axs2.set_aspect('equal', adjustable='box')
    # axs2.grid()
    # plt.show()

#i dati dentro points sono come segue:

# x_m,y_m,w_tr_right_m,w_tr_left_m
# 0.960975,4.022273,7.565,7.361

#il problema è che cambiando la rotazione cambia il concetto di sinistra e destra, ad esempio con NorisringFlippedX la rotazione di 60 gradi porta ad una rotazione tale che la destra e la sinistra causano un errore e sembra che il percorso sia di dimensioni piu larghe di quello che è