import os
import matplotlib.pyplot as plt
import numpy as np

def load_track_points(track_file):
    with open(track_file, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 4 and "#"!=line[0]:
            x, y, w_right, w_left = map(float, parts)
            points.append((x, y, w_right, w_left))
    return points

def load_racing_line_points(racing_line_file):
    with open(racing_line_file, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 2 and "#"!=line[0]:
            x, y = map(float, parts)
            points.append((x, y))
    return points

def load_racing_line_modified(racing_line_file):
    with open(racing_line_file, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 1 and "#"!=line[0]:
            w = float(parts[0])
            points.append(w)
    return points

def derivate(prec,p,succ):
    dx1 = p[0] - prec[0]
    dy1 = p[1] - prec[1]
    dx2 = succ[0] - p[0]
    dy2 = succ[1] - p[1]
    dx = (dx1 + dx2) / 2
    dy = (dy1 + dy2) / 2
    
    return dx,dy

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
            dx,dy=derivate(points[-1],points[0],points[1])
        elif i == len(points) - 1:
            dx,dy=derivate(points[i-1],points[i],points[0])
        else:
            dx,dy=derivate(points[i-1],points[i],points[i+1])

    # Vettore perpendicolare normalizzato
        perp = np.array([-dy, dx])
        perp /= np.linalg.norm(perp)
        normals.append(perp)
    # Calcolo bordo destro e sinistro
        cx, cy = points[i][0], points[i][1]
        wR, wL = points[i][2], points[i][3]
        w_right_x.append(cx - perp[0] * wR)
        w_right_y.append(cy - perp[1] * wR)
        w_left_x.append(cx + perp[0] * wL)
        w_left_y.append(cy + perp[1] * wL)
        center_x.append((w_right_x[-1] + w_left_x[-1]) / 2)
        center_y.append((w_right_y[-1] + w_left_y[-1]) / 2)

    normals=np.array(normals)
    w_right_x=np.array(w_right_x)
    w_right_y=np.array(w_right_y)
    w_left_x=np.array(w_left_x)
    center_x=np.array(center_x)
    center_y=np.array(center_y)
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
    cross = v1[1]*v2[0] - v1[0]*v2[1]
    # Calcola il prodotto scalare per l'angolo
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Evita divisione per zero
    angle = np.arccos(np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0))
    leeway = np.deg2rad(5)  # ad esempio 5 gradi di tolleranza
    if abs(angle) < leeway:
        return 0  # rettilineo
    if cross > 0:
        return -1  # sinistra
    elif cross < 0:
        return 1   # destra
    else:
        return 0   # rettilineo
    
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
        Ax = problem_side_x[j] - centerline[j][0]
        Ay = problem_side_y[j] - centerline[j][1]
        Bx = problem_side_x_modified[j] - centerline[j][0]
        By = problem_side_y_modified[j] - centerline[j][1]

        dotprod = np.dot([Ax,Ay],[Bx,By])
        lengtha = np.sqrt(Ax**2 + Ay**2)
        lengthb = np.sqrt(Bx**2 + By**2)

        #!!! radians !!!
        theta = np.arccos(dotprod / (lengtha * lengthb))
        thetas.append(theta)
            #normale nuova
        n=[(problem_side_x_modified[j]-centerline[j][0])/lengthb, (problem_side_y_modified[j]-centerline[j][1])/lengthb]
        new_normals.append(n)
       
        
        lunghezza_opposta=np.sqrt(np.power(opposite_side_x[j]-centerline[j][0],2) + np.power(opposite_side_y[j]-centerline[j][1],2)) 
        opposite_side_x_modified[j]=centerline[j][0]-n[0]*lunghezza_opposta
        opposite_side_y_modified[j]=centerline[j][1]-n[1]*lunghezza_opposta

    return problem_side_x_modified, problem_side_y_modified, opposite_side_x_modified, opposite_side_y_modified, thetas, new_normals

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def calc_intersect(A,B,C,D):
          #x1   x3     y3    y4
    numt=(A[0]-C[0])*(C[1]-D[1])-(A[1]-C[1])*(C[0]-D[0])
    dent=(A[0]-B[0])*(C[1]-D[1])-(A[1]-B[1])*(C[0]-D[0])
    t=numt/dent
    xt=A[0]+t*(B[0]-A[0])
    yt=A[1]+t*(B[1]-A[1])
    
    numu=(A[0]-B[0])*(A[1]-C[1])-(A[1]-B[1])*(A[0]-C[0])
    denu=(A[0]-B[0])*(C[1]-D[1])-(A[1]-B[1])*(C[0]-D[0])
    u=numu/denu
    
    return xt,yt,t

def add_points(prec,p1,p2,succ):
    dx1,dy1=derivate(prec,p1,p2)
    dx2,dy2=derivate(p1,p2,succ)

    p1n=[p1[0]+dx1,p1[1]+dy1]
    p2n=[p2[0]+dx2,p2[1]+dy2]
    x,y,t=calc_intersect(p1,p1n,p2,p2n)
    return x,y

if __name__ == "__main__":

    tracks_dir = "tracks/train/tracks"
    filename = "Norisring.csv"

    racing_line_dir = "tracks/train/racelines"
    
    track_path = os.path.join(tracks_dir, filename)
    racing_line_path= os.path.join(racing_line_dir, filename)

    points = load_track_points(track_path)
    raceline=load_racing_line_points(racing_line_path)
    
    right_x, right_y, left_x, left_y, center_x, center_y, normals = metodo2(points)
    fig, ax = plt.subplots(1, 1, figsize=(6,12))
    plot_tracks(ax,left_x, left_y, points, right_x, right_y, raceline,filename)
    


