#!/usr/bin/env python3
"""
Convert all PDFs in a folder to PNGs using PyMuPDF.
- Each PDF gets its own subfolder under the output directory.
- File naming: <pdf_name>_p001.png, <pdf_name>_p002.png, ...
"""

from __future__ import annotations
import argparse
from pathlib import Path
import fitz  # PyMuPDF
import tqdm
import os


def pdfs_to_png(
    input_dir: Path,
    out_dir: Path | None = None,
    dpi: int = 200,
    transparent: bool = False,
) -> None:
    input_dir = input_dir.expanduser().resolve()
    if out_dir is None:
        out_dir = "./png"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return

    scale = dpi / 72.0  # PDF point is 1/72 inch
    matrix = fitz.Matrix(scale, scale)

    for pdf_path in tqdm.tqdm(pdfs):
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"[SKIP] Failed to open {pdf_path.name}: {e}")
            continue

        if doc.is_encrypted:
            try:
                doc.authenticate("")  # try empty password
            except Exception:
                pass
            if doc.is_encrypted:
                print(f"[SKIP] Encrypted PDF (needs password): {pdf_path.name}")
                doc.close()
                continue

        # dest = out_dir / pdf_path.stem
        dest = f"{out_dir}/{pdf_path.stem}"
        if not os.path.exists(dest):
            os.makedirs(dest)
        # dest.mkdir(parents=True, exist_ok=True)

        # print(f"[PROCESS] {pdf_path.name} -> {dest}")
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=transparent)
            out_file = f"{dest}/{pdf_path.stem}_p{i+1:03d}.png"
            pix.save(str(out_file))
        doc.close()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert all PDFs in a folder to PNG images."
    )
    parser.add_argument("--inputs", type=Path, help="Folder containing PDFs")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder (default: <folder>/png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI (default: 200). Higher = bigger/clearer images.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Keep transparency (alpha channel). Default is opaque/flattened.",
    )

    args = parser.parse_args()
    pdfs_to_png(args.inputs, args.out, args.dpi, args.transparent)


if __name__ == "__main__":
    main()