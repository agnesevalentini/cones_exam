import os
import numpy as np
#from trajectory_planning_helpers.calc_normal_vectors import calc_normal_vectors
from trajectory_planning_helpers.check_normals_crossing import check_normals_crossing

def load_track_points(track_file):
    with open(track_file, "r") as f:
        lines = f.readlines()[1:]
    points = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 4:
            x, y, w_right, w_left = map(float, parts)
            points.append((x, y, w_right, w_left))
    return points

def load_racing_line_points(racing_line_file):
    with open(racing_line_file, "r") as f:
        lines = f.readlines()[1:]
    points = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            x, y = map(float, parts)
            points.append((x, y))
    return points

def metodo2(points):
    w_right_x=[]
    w_right_y=[]
    w_left_x=[]
    w_left_y=[]
    center_x=[]
    center_y=[]
    normals=[]
    # Calcola le direzioni tra i punti per disegnare le larghezze
    for i in range(len(points)):
        if i == 0:
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
        elif i == len(points) - 1:
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
        else:
            # Media dei vettori precedente e successivo
            dx1 = points[i][0] - points[i-1][0]
            dy1 = points[i][1] - points[i-1][1]
            dx2 = points[i+1][0] - points[i][0]
            dy2 = points[i+1][1] - points[i][1]
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2
    # Vettore perpendicolare normalizzato
        perp = np.array([-dy, dx])
        perp /= np.linalg.norm(perp)
        normals.append(perp)
    # Calcolo bordo destro e sinistro
        cx, cy = points[i][0], points[i][1]
        wR, wL = points[i][2], points[i][3]
        w_right_x.append(cx + perp[0] * wR)
        w_right_y.append(cy + perp[1] * wR)
        w_left_x.append(cx - perp[0] * wL)
        w_left_y.append(cy - perp[1] * wL)
        center_x.append((w_right_x[-1] + w_left_x[-1]) / 2)
        center_y.append((w_right_y[-1] + w_left_y[-1]) / 2)
        #crea center_x,center_y
    return w_right_x, w_right_y, w_left_x, w_left_y, center_x, center_y, normals

def plot_tracks_already_calced_normals(points, normals):
    w_right_x=[]

    w_right_y=[]
    w_left_x=[]
    w_left_y=[]
    center_x=[]
    center_y=[]
    for i in range(len(points)):
        perp = np.array([normals[i][0],normals[i][1]])
        cx, cy = points[i][0], points[i][1]
        wR, wL = points[i][2], points[i][3]
        w_right_x.append(cx - perp[0] * wR)
        w_right_y.append(cy - perp[1] * wR)
        w_left_x.append(cx + perp[0] * wL)
        w_left_y.append(cy + perp[1] * wL)
        center_x.append((w_right_x[-1] + w_left_x[-1]) / 2)
        center_y.append((w_right_y[-1] + w_left_y[-1]) / 2)
    return w_right_x, w_right_y, w_left_x, w_left_y, center_x, center_y

def plot_tracks(axs,left_x,left_y,centerline,right_x,right_y,raceline, filename):
    axs.plot(centerline[:,0], centerline[:,1], label=f"{filename} centerline", color='gray', linestyle='--')
    axs.plot(right_x, right_y, 'c', label="Metodo3 Right")
    axs.plot(raceline[:,0], raceline[:,1], 'yo-', label="best racing line")
    axs.plot(left_x, left_y, 'm', label="Metodo3 Left")
    for i in range(len(left_x)):
        xr, yr = right_x[i], right_y[i]
        xl, yl = left_x[i], left_y[i]
        axs.plot([xr, xl], [yr, yl], 'g--', linewidth=0.5)



def curva_direzione(prev_point, curr_point, next_point):
    # Vettore dal punto precedente al corrente
    v1 = np.array([curr_point[0] - prev_point[0], curr_point[1] - prev_point[1]])
    # Vettore dal corrente al successivo
    v2 = np.array([next_point[0] - curr_point[0], next_point[1] - curr_point[1]])
    # Calcola il prodotto vettoriale (solo la componente z)
    cross = v1[1]*v2[0] -v1[0]*v2[1]
    if cross > 0:
        return -1 #"sinistra"
    elif cross < 0:
        return  1 #"destra"
    else:
        return  0 #"rettilineo"
    
def correttore(problems, problem_side_x, problem_side_y, centerline, opposite_side_x, opposite_side_y):
    #sgarbuglio
    problems_modified=problems.copy()
    problem_side_x_modified=problem_side_x.copy()
    problem_side_y_modified=problem_side_y.copy()
    opposite_side_x_modified=opposite_side_x.copy()
    opposite_side_y_modified=opposite_side_y.copy()
    for i in problems_modified:
        closest=np.inf
        closest_indx=None
        for j in problems_modified:
            t=np.sqrt(np.power(problem_side_x_modified[j]-problem_side_x_modified[i-1],2) + np.power(problem_side_y_modified[j]-problem_side_y_modified[i-1],2))
            if(t<closest):
                closest=t
                closest_indx=j
        if closest_indx!=None:
            temp=problem_side_x_modified[i]
            problem_side_x_modified[i]=problem_side_x_modified[closest_indx]
            problem_side_x_modified[closest_indx]=temp
            
            temp=problem_side_y_modified[i]
            problem_side_y_modified[i]=problem_side_y_modified[closest_indx]
            problem_side_y_modified[closest_indx]=temp

        problems_modified.remove(i)
    
    #calcolo di quanto è stato modificato l'angolo
    thetas=[]
    new_normals=[]
    for j in problems:
                        #nuovo x               #centro         #vecchio x        #centro             #nuovo y              #centro              #vecchio y        #centro
        dotprod= (problem_side_x_modified[j]-centerline[j][0])*(problem_side_x[j]-centerline[j][0])+(problem_side_y_modified[i]-centerline[i][1])*(problem_side_y[i]-centerline[i][1])

        lengtha=np.sqrt(np.power(problem_side_x[j]-centerline[j][0],2) + np.power(problem_side_y[j]-centerline[j][1],2)) 
        lengthb=np.sqrt(np.power(problem_side_x_modified[j]-centerline[j][0],2) + np.power(problem_side_y_modified[j]-centerline[j][1],2)) 
        #TODO calcolare la lunghezza aggiornata di quel punto nuovo
        thetas.append(np.arccos(dotprod/(lengtha*lengthb)))
            #normale nuova
        n=((problem_side_x_modified[j]-centerline[j][0])/lengthb, (problem_side_y_modified[j]-centerline[j][1])/lengthb)
        new_normals.append(n)
       
        
        lunghezza_opposta=np.sqrt(np.power(opposite_side_x[j]-centerline[j][0],2) + np.power(opposite_side_y[j]-centerline[j][1],2)) 
        opposite_side_x_modified[j]=centerline[j][0]-n[0]*lunghezza_opposta
        opposite_side_y_modified[j]=centerline[j][1]-n[1]*lunghezza_opposta

    return problem_side_x_modified, problem_side_y_modified, opposite_side_x_modified, opposite_side_y_modified, thetas, new_normals


if __name__ == "__main__":

    tracks_dir = "tracks/train/tracks"
    filename = "Norisring.csv"

    racing_line_dir = "tracks/train/racelines"
    
    track_path = os.path.join(tracks_dir, filename)
    racing_line_path= os.path.join(racing_line_dir, filename)

    points = load_track_points(track_path)
    raceline=load_racing_line_points(racing_line_path)
    
    plot_tracks(points,raceline)
        


