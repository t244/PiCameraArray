"""Crop comparison over a whole burst: one camera vs mesh removed, one shared scale.

usage: crop_video.py <float dir> <out dir> x0 y0 w h [fps] [gamma]
"""
import glob, os, subprocess, sys
import cv2
import numpy as np

src, out = sys.argv[1], sys.argv[2]
x0, y0, cw, ch = (int(v) for v in sys.argv[3:7])
fps = float(sys.argv[7]) if len(sys.argv) > 7 else 10.0
G = float(sys.argv[8]) if len(sys.argv) > 8 else 0.75
frames = sorted(int(os.path.basename(p)[-8:-4]) for p in glob.glob(f"{src}/defenced_frame*.npy"))
crop = lambda a: a[y0:y0 + ch, x0:x0 + cw]
pairs = [(crop(np.load(f"{src}/reference_frame{k:04d}.npy")), crop(np.load(f"{src}/defenced_frame{k:04d}.npy")))
         for k in frames]
# one scale for every frame and both panels, so nothing flickers and nothing is favoured
pool = np.concatenate([np.concatenate([a.ravel()[::7], b.ravel()[::7]]) for a, b in pairs])
pool = pool[np.isfinite(pool)]
lo, hi = np.percentile(pool, 1.0), np.percentile(pool, 99.5)
m = lambda z: (np.clip((np.nan_to_num(z, nan=lo) - lo) / (hi - lo), 0, 1) ** G * 255).astype(np.uint8)
fdir = os.path.join(out, "crop_frames"); os.makedirs(fdir, exist_ok=True)
for n, (k, (a, b)) in enumerate(zip(frames, pairs)):
    img = np.hstack([m(a), np.full((ch, 16), 255, np.uint8), m(b)])
    img = cv2.copyMakeBorder(img, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
    for x, t in ((8, "one camera"), (cw + 24, "mesh removed")):
        cv2.putText(img, t, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    cv2.putText(img, f"frame {k:02d}  ({k / 30:.2f} s)", (2 * cw - 210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 80, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(fdir, f"{n:04d}.png"), img)
mp4 = os.path.join(out, f"crop_compare_x{x0}_y{y0}.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(fdir, "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", mp4], check=True)
print(f"{len(frames)} frames, shared range {lo:.1f}..{hi:.1f}, gamma {G}, {fps} fps -> {mp4}")
