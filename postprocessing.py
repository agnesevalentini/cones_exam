#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Dict


PREFERRED_EXT_ORDER = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


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
            "Aggiunge l'estensione ai file in 'images' usando quelle trovate in 'img' "
            "(match per basename)."
        )
    )
    parser.add_argument(
        "src_root",
        help="Cartella sorgente che contiene la sottocartella 'img' con i file con estensione",
    )
    parser.add_argument(
        "dst_root",
        help="Cartella destinazione che contiene la sottocartella 'images' con file senza estensione",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    src_img = Path(args.src_root).expanduser().resolve() / "img"
    dst_images = Path(args.dst_root).expanduser().resolve() / "images"

    if not src_img.is_dir():
        print(f"Errore: la cartella 'img' non esiste in: {src_img.parent}")
        return 1
    if not dst_images.is_dir():
        print(f"Errore: la cartella 'images' non esiste in: {dst_images.parent}")
        return 1

    print(f"Sorgente (img): {src_img}")
    print(f"Destinazione (images): {dst_images}")

    mapping = build_extension_map(src_img, case_insensitive=args.case_insensitive)
    print(f"Trovate {len(mapping)} associazioni basename -> estensione dalla sorgente.")

    renamed, skipped, missing = rename_images(
        dst_images, mapping, dry_run=args.dry_run, case_insensitive=args.case_insensitive
    )

    print("\nRiepilogo:")
    print(f"  Rinomine effettuate: {renamed}")
    print(f"  File già con estensione o target esistente: {skipped}")
    print(f"  Basename senza match in sorgente: {missing}")

    if args.dry_run:
        print("\nNota: era una simulazione (dry-run), nessun file è stato modificato.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
