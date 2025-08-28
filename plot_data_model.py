import matplotlib.pyplot as plt
import numpy as np
import os

    # Funzione per aggiungere massimo con linea ed etichetta
def mark_max(ax, x, y, label):
    max_idx = np.argmax(y)
    ax.axvline(x[max_idx], color="red", linestyle="--", alpha=0.7)  # linea verticale
    ax.scatter(x[max_idx], y[max_idx], color="red")                 # punto massimo
    ax.text(
        x[max_idx], y[max_idx],
        f"{y[max_idx]:.3f}", 
        color="red", ha="left", va="bottom", fontsize=9
    )
    ax.set_title(label)

def mark_min(ax, x, y, label):
    max_idx = np.argmin(y)
    ax.axvline(x[max_idx], color="red", linestyle="--", alpha=0.7)  # linea verticale
    ax.scatter(x[max_idx], y[max_idx], color="red")                 # punto massimo
    ax.text(
        x[max_idx], y[max_idx],
        f"{y[max_idx]:.3f}", 
        color="red", ha="left", va="bottom", fontsize=9
    )
    ax.set_title(label)



# Carica i dati dal file CSV (senza header)
model_dir="data/v2/"
datafilenames =np.array( [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".csv")])
datafilenames=sorted(datafilenames)
for i in range(0,len(datafilenames),2):
    data_test = np.loadtxt(model_dir+datafilenames[i], delimiter=",")
    data_train = np.loadtxt(model_dir+datafilenames[i+1], delimiter=",")
    # Ogni colonna corrisponde a una metrica
    loss_hist = data_test[:, 0]
    r2_hist   = data_test[:, 1]
    pois_hist = data_test[:, 2]

    loss_train_hist = data_train[:, 0]
    r2_train_hist   = data_train[:, 1]
    pois_train_hist = data_train[:, 2]

    # Creiamo la variabile "epoche"
    epochs = np.arange(1, len(loss_hist) + 1)

    plt.figure(figsize=(12, 6))
    plt.suptitle(datafilenames[i]+"\n"+datafilenames[i+1])
    # Loss
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(epochs, loss_hist, label="Loss Test")
    ax1.plot(epochs, loss_train_hist, label="Loss Train")
    mark_min(ax1, epochs, loss_hist, "Loss (Test)")
    mark_min(ax1, epochs, loss_train_hist, "Loss (Train)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # R²
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(epochs, r2_hist, label="R² Test")
    ax2.plot(epochs, r2_train_hist, label="R² Train")
    mark_max(ax2, epochs, r2_hist, "R² (Test)")
    mark_max(ax2, epochs, r2_train_hist, "R² (Train)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R²")
    ax2.legend()

    # Poisson
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(epochs, pois_hist, label="Poisson Test")
    ax3.plot(epochs, pois_train_hist, label="Poisson Train")
    mark_min(ax3, epochs, pois_hist, "Poisson (Test)")
    mark_min(ax3, epochs, pois_train_hist, "Poisson (Train)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Poisson Loss")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(model_dir+datafilenames[i].removesuffix('test.csv')+".png",format="png")
