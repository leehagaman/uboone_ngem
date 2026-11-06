#!/usr/bin/env python3
"""Arrange a multi-page PDF into a single-page grid (default 4 rows x 5 cols).

Each input page is rotated 90 CW (to bring landscape ROOT plots upright) and
optionally trimmed so cells abut tightly. --trim takes a single value or four
comma-separated values L,B,R,T in PDF points (1pt = 1/72 in), expressed in
the FINAL upright orientation (so L = left side of the plot you see).

Caveat: --trim works cleanly when source pages do not fill an opaque
background. If you see one cell's background wiping a neighbor, pre-crop with
`pdfcrop input.pdf cropped.pdf` (texlive) and run with --trim 0.
"""

import sys
import os
import subprocess
import argparse
import pypdf


def parse_trim(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) == 1:
        return (parts[0],) * 4
    if len(parts) == 4:
        return tuple(parts)
    raise argparse.ArgumentTypeError("--trim takes 1 or 4 comma-separated numbers")


def make_grid(input_path, output_path, rows, cols, trim, box):
    reader = pypdf.PdfReader(input_path)
    n = rows * cols
    if len(reader.pages) < n:
        raise ValueError(f"Input PDF has {len(reader.pages)} pages, need {n}.")

    def get_box(page):
        b = page.cropbox if box == "crop" else page.mediabox
        return (float(b.left), float(b.bottom), float(b.right), float(b.top))

    bl, bb, br, bt = get_box(reader.pages[0])
    src_w = br - bl
    src_h = bt - bb

    # Rotation is fixed at 90 CW. Map final-orientation trim to source trim:
    # final L←source B, final B←source R, final R←source T, final T←source L.
    fL, fB, fR, fT = trim
    sL, sB, sR, sT = fT, fL, fB, fR

    box_w_src = src_w - sL - sR
    box_h_src = src_h - sB - sT
    if box_w_src <= 0 or box_h_src <= 0:
        raise ValueError("Trim values exceed page size.")

    cell_w = box_h_src
    cell_h = box_w_src

    writer = pypdf.PdfWriter()
    out_page = writer.add_blank_page(width=cell_w * cols, height=cell_h * rows)

    for i in range(n):
        r = i // cols
        c = i % cols
        cx = c * cell_w
        cy = (rows - 1 - r) * cell_h

        pl, pb, _, _ = get_box(reader.pages[i])
        t = (pypdf.Transformation()
             .translate(-(pl + sL), -(pb + sB))
             .rotate(270)
             .translate(0, box_w_src)
             .translate(cx, cy))

        out_page.merge_transformed_page(reader.pages[i], t)

    with open(output_path, "wb") as f:
        writer.write(f)


def render_jpeg(pdf_path, jpeg_path, dpi):
    prefix, _ = os.path.splitext(jpeg_path)
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", str(dpi), "-singlefile", pdf_path, prefix],
        check=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--trim", type=parse_trim, default=(0, 0, 0, 0),
                   help="Trim margins in points (final orientation): "
                        "single value or L,B,R,T")
    p.add_argument("--box", choices=("crop", "media"), default="crop",
                   help="Source page extent to use (default: crop)")
    p.add_argument("--jpeg-dpi", type=int, default=200,
                   help="Resolution for the companion JPEG (default: 200)")
    args = p.parse_args()

    make_grid(args.input, args.output, args.rows, args.cols,
              args.trim, args.box)

    base, ext = os.path.splitext(args.output)
    jpeg_path = (base if ext.lower() == ".pdf" else args.output) + ".jpg"
    render_jpeg(args.output, jpeg_path, args.jpeg_dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
