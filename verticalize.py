import pandas as pd

# Leggi il CSV originale (senza header, separatore = virgola)
df = pd.read_csv("experimentaldata_select_best_model_lr0,003train.csv", header=None)

# Trasponi il DataFrame (righe <-> colonne)
df_T = df.T

# Salva il nuovo file
df_T.to_csv("experimentaldata_select_best_model_lr0,003train.csv", index=False, header=False)

print("File trasposto salvato")