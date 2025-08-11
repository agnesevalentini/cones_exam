#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List
import random


PREFERRED_EXT_ORDER = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
LABEL_EXTS = [".json", ".txt"]


def ext_priority(ext: str) -> int:
    ext = ext.lower()
    try:
        return len(PREFERRED_EXT_ORDER) - PREFERRED_EXT_ORDER.index(ext)
    except ValueError:
        # Unknown extensions get lowest priority
        return 0


def build_extension_map(img_dir: Path, case_insensitive: bool = False) -> Dict[str, str]:
    """Build a map: basename -> extension from files in img_dir.

    If multiple files share the same basename but different extensions, prefer
    extensions by PREFERRED_EXT_ORDER.
    """
    mapping: Dict[str, str] = {}
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        if not p.suffix:
            # Skip files without extension in source
            continue
        stem = p.stem
        key = stem.lower() if case_insensitive else stem
        ext = p.suffix  # includes leading dot
        if key not in mapping:
            mapping[key] = ext
        else:
            # Choose by priority
            if ext_priority(ext) > ext_priority(mapping[key]):
                mapping[key] = ext
    return mapping


def build_extension_map_multi(img_dirs: Iterable[Path], case_insensitive: bool = False) -> Dict[str, str]:
    """Unisce le mappe di estensioni provenienti da più cartelle 'img'.

    In caso di conflitto sullo stesso basename, sceglie l'estensione con
    priorità più alta secondo PREFERRED_EXT_ORDER (indipendente dall'ordine
    delle sorgenti).
    """
    combined: Dict[str, str] = {}
    for img_dir in img_dirs:
        partial = build_extension_map(img_dir, case_insensitive=case_insensitive)
        for key, ext in partial.items():
            if key not in combined or ext_priority(ext) > ext_priority(combined[key]):
                combined[key] = ext
    return combined


def rename_images(images_dir: Path, mapping: Dict[str, str], dry_run: bool = False, case_insensitive: bool = False) -> tuple[int, int, int]:
    """Rename files in images_dir by appending the extension found in mapping.

    Returns a tuple: (renamed_count, skipped_with_ext_count, missing_in_map_count)
    """
    renamed = 0
    skipped_already_ext = 0
    missing = 0

    for f in images_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix:
            skipped_already_ext += 1
            print(f"[SKIP] Ha già estensione: {f.name}")
            continue

        base = f.name  # name without extension (expected)
        key = base.lower() if case_insensitive else base
        ext = mapping.get(key)
        if not ext:
            missing += 1
            print(f"[WARN] Nessuna estensione trovata per '{f.name}'")
            continue

        target = f.with_name(base + ext)
        if target.exists():
            # If target already exists, avoid overwriting
            print(f"[SKIP] Esiste già il file di destinazione: {target.name}")
            skipped_already_ext += 1
            continue

        if dry_run:
            print(f"[DRY-RUN] Rinominerei: {f.name} -> {target.name}")
        else:
            f.rename(target)
            print(f"[OK] Rinominato: {f.name} -> {target.name}")
        renamed += 1

    return renamed, skipped_already_ext, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggiunge l'estensione ai file in 'images' usando quelle trovate in una o più "
            "cartelle sorgenti contenenti 'img' (match per basename)."
        )
    )
    parser.add_argument(
        "src_root",
        help="Cartella sorgente principale che contiene la sottocartella 'img' con i file con estensione",
    )
    parser.add_argument(
        "dst_root",
        help="Cartella destinazione che contiene la sottocartella 'images' con file senza estensione",
    )
    parser.add_argument(
        "-s",
        "--src",
        dest="extra_src_roots",
        action="append",
        default=[],
        help=(
            "Cartella sorgente AGGIUNTIVA (ripetibile) che contiene la sottocartella 'img'. "
            "Le estensioni verranno aggregate tra tutte le sorgenti."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra le rinomine senza applicarle",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Confronta i nomi senza distinzione tra maiuscole/minuscole",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Esegue uno split random 50/50 dei file in images e dei corrispondenti JSON in labels "
            "in sottocartelle train/val (non ricorsivo)."
        ),
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.5,
        help="Frazione dei file da mettere in train (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed per la randomizzazione dello split",
    )
    return parser.parse_args()


def split_train_val(
    images_dir: Path,
    labels_dir: Path,
    ratio: float = 0.5,
    seed: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Esegue split random dei file immagine (solo top-level) in images/train e images/val.

    Per ogni immagine spostata, se esiste un JSON in labels con lo stesso basename,
    sposta anche quello in labels/train o labels/val.

    Ritorna un dizionario con conteggi riassuntivi.
    """
    if ratio <= 0 or ratio >= 1:
        print(f"[WARN] split-ratio={ratio} non valido; uso 0.5")
        ratio = 0.5

    # Considera file immagine nella cartella principale (no sottocartelle):
    # - con estensione nota
    # - oppure senza estensione (per supportare workflow pre-rename)
    candidates = []
    for f in images_dir.iterdir():
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf in PREFERRED_EXT_ORDER or suf == "":
            candidates.append(f)

    total = len(candidates)
    if total == 0:
        print("[INFO] Nessuna immagine da suddividere in 'images'.")
        return {
            "total": 0,
            "train_images": 0,
            "val_images": 0,
            "train_labels": 0,
            "val_labels": 0,
            "missing_labels": 0,
            "skipped_existing": 0,
        }

    rng = random.Random(seed)
    rng.shuffle(candidates)
    split_idx = int(total * ratio)
    train_set = candidates[:split_idx]
    val_set = candidates[split_idx:]

    # Prepara directory di destinazione
    images_train = images_dir / "train"
    images_val = images_dir / "val"
    labels_train = labels_dir / "train"
    labels_val = labels_dir / "val"

    for d in [images_train, images_val, labels_dir, labels_train, labels_val]:
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": total,
        "train_images": 0,
        "val_images": 0,
        "train_labels": 0,
        "val_labels": 0,
        "missing_labels": 0,
        "skipped_existing": 0,
    }

    def move_pair(img_path: Path, dest_img_dir: Path, dest_lbl_dir: Path) -> None:
        nonlocal stats
        dst_img = dest_img_dir / img_path.name
        if dst_img.exists():
            print(f"[SKIP] File di destinazione immagine esiste già: {dst_img}")
            stats["skipped_existing"] += 1
        else:
            if dry_run:
                print(f"[DRY-RUN] Sposterei img: {img_path} -> {dst_img}")
            else:
                img_path.rename(dst_img)
            # Aggiorna conteggio immagini
            if dest_img_dir is images_train:
                stats["train_images"] += 1
            else:
                stats["val_images"] += 1

        # Gestisci label (supporta .json e .txt). Considera label mancante se nessuna delle estensioni esiste
        found_any_label = False
        for ext in LABEL_EXTS:
            lbl_src = labels_dir / (img_path.stem + ext)
            if not lbl_src.exists():
                continue
            found_any_label = True
            dst_lbl = dest_lbl_dir / lbl_src.name
            if dst_lbl.exists():
                print(f"[SKIP] File di destinazione label esiste già: {dst_lbl}")
                stats["skipped_existing"] += 1
                continue
            if dry_run:
                print(f"[DRY-RUN] Sposterei lbl: {lbl_src} -> {dst_lbl}")
            else:
                lbl_src.rename(dst_lbl)
            if dest_lbl_dir is labels_train:
                stats["train_labels"] += 1
            else:
                stats["val_labels"] += 1
        if not found_any_label:
            stats["missing_labels"] += 1
            print(f"[WARN] Label mancante per: {img_path.stem}")

    for img in train_set:
        move_pair(img, images_train, labels_train)
    for img in val_set:
        move_pair(img, images_val, labels_val)

    return stats


def main() -> int:
    args = parse_args()

    # Colleziona tutte le cartelle 'img' dalle sorgenti specificate
    src_roots: List[Path] = [Path(args.src_root)] + [Path(p) for p in args.extra_src_roots]
    src_imgs: List[Path] = [(p.expanduser().resolve() / "img") for p in src_roots]
    dst_images = Path(args.dst_root).expanduser().resolve() / "images"

    # Filtra solo quelle esistenti e segnala quelle mancanti
    existing_src_imgs = []
    missing_src_imgs = []
    for d in src_imgs:
        if d.is_dir():
            existing_src_imgs.append(d)
        else:
            missing_src_imgs.append(d)

    if missing_src_imgs:
        for d in missing_src_imgs:
            print(f"[WARN] La cartella 'img' non esiste: {d}")

    if not existing_src_imgs:
        print("Errore: nessuna cartella sorgente valida trovata (con sottocartella 'img').")
        return 1
    if not dst_images.is_dir():
        print(f"Errore: la cartella 'images' non esiste in: {dst_images.parent}")
        return 1

    print("Sorgenti (img):")
    for d in existing_src_imgs:
        print(f"  - {d}")
    print(f"Destinazione (images): {dst_images}")

    mapping = build_extension_map_multi(existing_src_imgs, case_insensitive=args.case_insensitive)
    print(f"Trovate {len(mapping)} associazioni basename -> estensione dalle sorgenti.")

    renamed, skipped, missing = rename_images(
        dst_images, mapping, dry_run=args.dry_run, case_insensitive=args.case_insensitive
    )

    print("\nRiepilogo:")
    print(f"  Rinomine effettuate: {renamed}")
    print(f"  File già con estensione o target esistente: {skipped}")
    print(f"  Basename senza match in sorgente: {missing}")

    # Esegui split train/val se richiesto
    if args.split:
        labels_dir = Path(args.dst_root).expanduser().resolve() / "labels"
        if not labels_dir.exists() and not args.dry_run:
            labels_dir.mkdir(parents=True, exist_ok=True)
        print("\nEseguo split train/val...")
        split_stats = split_train_val(
            dst_images,
            labels_dir,
            ratio=args.split_ratio,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        print("\nRiepilogo split:")
        print(f"  Totale immagini considerate: {split_stats['total']}")
        print(f"  Immagini in train: {split_stats['train_images']}")
        print(f"  Immagini in val:   {split_stats['val_images']}")
        print(f"  Label JSON in train: {split_stats['train_labels']}")
        print(f"  Label JSON in val:   {split_stats['val_labels']}")
        print(f"  Label mancanti: {split_stats['missing_labels']}")
        if split_stats["skipped_existing"]:
            print(f"  Spostamenti saltati per esistenza target: {split_stats['skipped_existing']}")

    if args.dry_run:
        print("\nNota: era una simulazione (dry-run), nessun file è stato modificato.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
