import os
from pathlib import Path
import shutil


input_folders = input("Folders to preprocess in a format accepted by fscoco tools, space separated. \n Remember that the meta.json must be at the same level of all folders: ")
input_folders = input_folders.split()

project_folder = input("Project folder name (will be created in current directory): ")

dataset_dir = Path(project_folder) / "dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)

# Copia meta.json dal livello superiore dentro la cartella project_folder
meta_src = Path(input_folders[0]).parent / "meta.json"
meta_dst = Path(project_folder) / "meta.json"
if meta_src.exists():
    if meta_dst.exists():
        print(f"Avviso: 'meta.json' già presente in {meta_dst}, salto la copia.")
    else:
        shutil.copy2(meta_src, meta_dst)
else:
    print(f"Attenzione: 'meta.json' non trovato in {meta_src}")

for folder in input_folders:
    for name in ("ann", "img"):
        src = Path(folder) / name
        dest = dataset_dir / name
        if not src.exists():
            print(f"Attenzione: sorgente non trovata: {src}")
        else:
            print(f"Copia da {src} a {dest} di {name}")
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

# # ANN: rimuovere l'estensione in mezzo prima di .json se presente (es. .jpg.json -> .json)
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

# Sincronizzazione: elimina file senza corrispondente tra img e ann (ignorando estensioni)
if img_dir.exists() and ann_dir.exists():
    # Costruisci insiemi di percorsi relativi senza estensione
    img_bases = set()
    for p in img_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(img_dir)
        # rimuovi tutti i suffissi (robusto anche se non ce ne sono)
        while rel.suffix:
            rel = rel.with_suffix("")
        img_bases.add(rel.as_posix())

    ann_bases = set()
    for p in ann_dir.rglob("*.json"):
        rel = p.relative_to(ann_dir)
        rel = rel.with_suffix("")  # togli .json
        # nel caso residuassero estensioni di immagine
        while rel.suffix.lower() in img_exts:
            rel = rel.with_suffix("")
        ann_bases.add(rel.as_posix())

    # Elimina annotazioni senza immagine corrispondente
    for p in ann_dir.rglob("*.json"):
        rel = p.relative_to(ann_dir)
        rel = rel.with_suffix("")
        while rel.suffix.lower() in img_exts:
            rel = rel.with_suffix("")
        if rel.as_posix() not in img_bases:
            print(f"Rimuovo annotation senza immagine: {p}")
            p.unlink()

    # Elimina immagini senza annotazione corrispondente
    for p in img_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(img_dir)
        while rel.suffix:
            rel = rel.with_suffix("")
        if rel.as_posix() not in ann_bases:
            print(f"Rimuovo immagine senza annotation: {p}")
            p.unlink()


