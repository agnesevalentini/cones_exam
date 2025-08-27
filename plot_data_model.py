import matplotlib.pyplot as plt
import numpy as np

# Carica i dati dal file CSV (senza header)
data_test = np.loadtxt("select_best_model_and_lr_stable_no_dist_1_models_lr0,003test.csv", delimiter=",")
data_train = np.loadtxt("select_best_model_and_lr_stable_no_dist_1_models_lr0,003train.csv", delimiter=",")

# Ogni colonna corrisponde a una metrica
loss_hist = data_test[:, 0]
r2_hist   = data_test[:, 1]
pois_hist = data_test[:, 2]

loss_train_hist = data_train[:, 0]
r2_train_hist   = data_train[:, 1]
pois_train_hist = data_train[:, 2]

# Creiamo la variabile "epoche"
epochs = np.arange(1, len(loss_hist) + 1)

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

plt.figure(figsize=(12, 6))

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
plt.show()
