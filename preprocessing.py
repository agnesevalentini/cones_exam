import os
from pathlib import Path
import shutil


project_folder = input("Folder to preprocess in a format accepted by fscoco tools: ")
dataset_dir = Path(project_folder) / "dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)

# Copia meta.json dal livello superiore dentro la cartella project_folder
meta_src = Path(project_folder).parent / "meta.json"
meta_dst = Path(project_folder) / "meta.json"
if meta_src.exists():
    if meta_dst.exists():
        print(f"Avviso: 'meta.json' già presente in {meta_dst}, salto la copia.")
    else:
        shutil.copy2(meta_src, meta_dst)
else:
    print(f"Attenzione: 'meta.json' non trovato in {meta_src}")

for name in ("ann", "img"):
    src = Path(project_folder) / name
    dest = dataset_dir / name
    if not src.exists():
        print(f"Attenzione: sorgente non trovata: {src}")
        continue
    shutil.copytree(src, dest, dirs_exist_ok=True)

# Normalizzazione nomi file
img_dir = dataset_dir / "img"
ann_dir = dataset_dir / "ann"

# IMG: rimuovere tutte le estensioni (nome senza suffissi)
if img_dir.exists():
    for p in img_dir.rglob("*"):
        if not p.is_file():
            continue
        q = p
        while q.suffix:
            q = q.with_suffix("")
        if q == p:
            continue
        if q.exists():
            print(f"Attenzione: file di destinazione già esistente, salto: {q}")
            continue
        p.rename(q)

# ANN: rimuovere l'estensione in mezzo prima di .json se presente (es. .jpg.json -> .json)
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
if ann_dir.exists():
    for p in ann_dir.rglob("*.json"):
        base = p.with_suffix("")  # rimuove .json
        while base.suffix.lower() in img_exts:
            base = base.with_suffix("")
        q = base.with_suffix(".json")
        if q == p:
            continue
        if q.exists():
            print(f"Attenzione: file di destinazione già esistente, salto: {q}")
            continue
        p.rename(q)


