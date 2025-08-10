import os
import matplotlib.pyplot as plt
import numpy as np

tracks_dir = "tracks/train/tracks"

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

def metodo1(points):
    w_right_x=[]
    w_right_y=[]
    w_left_x=[]
    w_left_y=[]
    center_x=[]
    center_y=[]

    dx=points[1][0]-points[0][0]
    dy=points[1][1]-points[0][1]
    perp = np.array([-dy, dx])
    perp /= np.linalg.norm(perp)
    w_right_x.append(points[0][0] + perp[0] * points[0][2])
    w_right_y.append(points[0][1] + perp[1] * points[0][2])
    w_left_x.append(points[0][0] - perp[0] * points[0][3])
    w_left_y.append(points[0][1] - perp[1] * points[0][3])
    #derived centerline
    # Calcola la centerline come punto medio tra bordo destro e sinistro
    center_x.append((w_right_x[0] + w_left_x[0]) / 2)
    center_y.append((w_right_y[0] + w_left_y[0]) / 2)
    
    
    for i in range(1,len(points)):
        
        x,y,w_right, w_left = points[i]

        dx=x-points[i-1][0]
        dy=y-points[i-1][1]

        perp = np.array([-dy, dx])
        perp /= np.linalg.norm(perp)

        w_right_x.append(x + perp[0] * w_right)
        w_right_y.append(y + perp[1] * w_right)
        w_left_x.append(x - perp[0] * w_left)
        w_left_y.append(y - perp[1] * w_left)
        #derived centerline
        center_x.append((w_right_x[-1] + w_left_x[-1]) / 2)
        center_y.append((w_right_y[-1] + w_left_y[-1]) / 2)
    
    return w_right_x, w_right_y, w_left_x, w_left_y, center_x, center_y

def metodo2(points):
    w_right_x=[]
    w_right_y=[]
    w_left_x=[]
    w_left_y=[]
    center_x=[]
    center_y=[]
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
    return w_right_x, w_right_y, w_left_x, w_left_y, center_x, center_y

def plot_tracks(tracks_dir):
    filename = "Norisring.csv"
    if filename.endswith(".csv"):
        track_path = os.path.join(tracks_dir, filename)
        points = load_track_points(track_path)
        if points:
            xs, ys, _, _ = zip(*points)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Metodo 1
            w_right_x1, w_right_y1, w_left_x1, w_left_y1, center_x1, center_y1 = metodo1(points)
            axs[0].plot(xs, ys, label=f"{filename} centerline", color='gray', linestyle='--')
            axs[0].plot(w_right_x1, w_right_y1, 'b-', label="Metodo1 Right")
            axs[0].plot(w_left_x1, w_left_y1, 'r-', label="Metodo1 Left")
            axs[0].plot(center_x1, center_y1, 'ko', markersize=2, label="Metodo1 Center")
            axs[0].set_title("Metodo1")
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")
            axs[0].axis('equal')
            axs[0].legend()

            # Metodo 2
            w_right_x2, w_right_y2, w_left_x2, w_left_y2, center_x2, center_y2 = metodo2(points)
            axs[1].plot(xs, ys, label=f"{filename} centerline", color='gray', linestyle='--')
            axs[1].plot(w_right_x2, w_right_y2, 'c', label="Metodo2 Right")
            axs[1].plot(w_left_x2, w_left_y2, 'm', label="Metodo2 Left")
            axs[1].plot(center_x2, center_y2, 'go', markersize=2, label="Metodo2 Center")
            axs[1].set_title("Metodo2")
            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Y")
            axs[1].axis('equal')
            axs[1].legend()

            plt.suptitle("Racetrack Points: Metodo1 vs Metodo2")
            plt.tight_layout()
            plt.show()




if __name__ == "__main__":
    plot_tracks(tracks_dir)

