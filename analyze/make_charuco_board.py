#!/usr/bin/env python3
"""Draw the ChArUco calibration board as a vector A4 PDF, at true size.

The board matches analyze/calibrate.py exactly (change both together):

    dictionary   DICT_4X4_100
    squares      5 x 7 (cols x rows), portrait
    square       38.0 mm
    marker       20.5 mm

Squares and marker cells are drawn as filled rectangles, so the print is sharp
at any printer resolution. Which squares are black and which marker id sits in
which square are read from OpenCV's own CharucoBoard, not re-derived here.

Print at 100 % ("actual size", no fit-to-page) and check the 100 mm scale bar
with a ruler; calibrate.py assumes the sizes above. A JSON file with the same
parameters is written next to the PDF.

    python make_charuco_board.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath
import numpy as np

DICT_NAME = "DICT_4X4_100"
SQUARES = (5, 7)            # (cols, rows) as cv2.aruco.CharucoBoard takes them
SQUARE_MM = 38.0
MARKER_MM = 20.5
PAGE_MM = (210.0, 297.0)    # A4 portrait


def board():
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    return d, cv2.aruco.CharucoBoard(SQUARES, SQUARE_MM / 1000, MARKER_MM / 1000, d)


def draw(path):
    d, b = board()
    cols, rows = SQUARES
    W, H = cols * SQUARE_MM, rows * SQUARE_MM
    x0, y0 = (PAGE_MM[0] - W) / 2, (PAGE_MM[1] - H) / 2      # board top-left, mm

    # Black squares: sample OpenCV's own rendering at each square centre.
    px = 20
    ref = b.generateImage((cols * px, rows * px), marginSize=0, borderBits=1)
    black = [(c, r) for r in range(rows) for c in range(cols)
             if ref[r * px + 1, c * px + 1] < 128]

    fig = plt.figure(figsize=(PAGE_MM[0] / 25.4, PAGE_MM[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_MM[0])
    ax.set_ylim(PAGE_MM[1], 0)                               # y down, in mm
    ax.set_axis_off()

    def box(x, y, w, h, colour="black"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, edgecolor="none",
                               linewidth=0, antialiased=False))

    for c, r in black:
        box(x0 + c * SQUARE_MM, y0 + r * SQUARE_MM, SQUARE_MM, SQUARE_MM)

    # Markers: corners from the board's object points (metres, y down),
    # cells from the dictionary bitmap (1-cell black border included).
    nbits = d.markerSize + 2
    cell = MARKER_MM / nbits
    for ids, pts in zip(np.asarray(b.getIds()).ravel(), b.getObjPoints()):
        mx, my = x0 + pts[0][0] * 1000, y0 + pts[0][1] * 1000
        bits = cv2.aruco.generateImageMarker(d, int(ids), nbits, borderBits=1)
        # One black square, then all white cells as ONE compound path: adjacent
        # cells drawn as separate rectangles leave hairline seams in the print.
        box(mx, my, MARKER_MM, MARKER_MM)
        verts, codes = [], []
        for i in range(nbits):
            for j in range(nbits):
                if bits[i, j] >= 128:
                    x, y = mx + j * cell, my + i * cell
                    verts += [(x, y), (x + cell, y), (x + cell, y + cell), (x, y + cell), (x, y)]
                    codes += [MPath.MOVETO] + [MPath.LINETO] * 3 + [MPath.CLOSEPOLY]
        if verts:
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor="white",
                                   edgecolor="none", linewidth=0, antialiased=False))

    # Label and a 100 mm scale bar in the bottom margin.
    ty = y0 + H + 5
    ax.text(x0, ty, f"ChArUco  {DICT_NAME}  {cols}x{rows} squares  "
            f"square {SQUARE_MM:g} mm  marker {MARKER_MM:g} mm  "
            "(print at 100 %, check scale bar)",
            fontsize=6.5, va="top", ha="left", family="DejaVu Sans")
    box(x0, ty + 4, 100.0, 0.6)
    for k in range(0, 101, 10):
        box(x0 + k - 0.15, ty + 3 if k % 50 == 0 else ty + 3.6, 0.3, 1.0 if k % 50 == 0 else 0.4)
    ax.text(x0 + 102, ty + 4.3, "100 mm", fontsize=6, va="center", ha="left")

    fig.savefig(path)
    plt.close(fig)
    return (x0, y0)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_A4_{DICT_NAME}_{SQUARES[0]}x{SQUARES[1]}_{SQUARE_MM:g}mm_{MARKER_MM:g}mm"
    pdf = out / f"{stem}.pdf"
    x0, y0 = draw(pdf)
    meta = {"dictionary": DICT_NAME, "squares_x": SQUARES[0], "squares_y": SQUARES[1],
            "square_length_mm": SQUARE_MM, "marker_length_mm": MARKER_MM,
            "page": "A4 portrait 210x297 mm, print at 100 %",
            "board_offset_on_page_mm": [x0, y0],
            "opencv": cv2.__version__,
            "opencv_call": f"cv2.aruco.CharucoBoard(({SQUARES[0]}, {SQUARES[1]}), "
                           f"{SQUARE_MM / 1000}, {MARKER_MM / 1000}, "
                           f"cv2.aruco.getPredefinedDictionary(cv2.aruco.{DICT_NAME}))",
            "used_by": "analyze/calibrate.py"}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(pdf)


if __name__ == "__main__":
    main()
