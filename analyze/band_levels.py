#!/usr/bin/env python3
"""
band_levels.py - per-burst, per-frame brightness statistics of every camera,
raw and warped onto one common view, for the moisture analysis.

This is the measurement half of the moisture analysis, kept deliberately dumb:
it reads frames and writes numbers, and makes no choice that the analysis
might want to revisit. Which bursts, how many frames, which statistic, which
cameras make a band - all of that is decided afterwards by
analyze/moisture_floor.py from the json lines written here, so trying another
choice costs seconds instead of another pass over 3 TB.

PER FRAME, PER CAMERA
    raw     statistics of the camera's own image inside --roi
    warp    the same after warping onto the --ref camera at --z, inside the
            same --roi of the REFERENCE view, so every camera looks at the
            same patch of the drum
PER FRAME, PER BAND
    band    statistics of the band's views averaged pixel by pixel after the
            warp, each view first divided by its fixed gain (--gains, or the
            median ratio measured on the first bursts processed)

Each statistic is [mean, p50, p90, p95, p99] of black-subtracted 10-bit counts.

The fence de-fencing stage is not run: on the bursts it was measured on it
moved the floor from 5.06 % to 5.02 %, and it costs ~15 s per band per frame.

USAGE
    python analyze/band_levels.py <packed session dir> --calib calibration_results.npz \\
        --out levels.jsonl [--every 1] [--frames 0:90:3] [--workers 8]

Re-running with the same --out skips the bursts already written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sa_packed as S                                   # noqa: E402

STATS = ("mean", "p50", "p90", "p95", "p99")
_CTX = {}


def stats(a: np.ndarray):
    # every other pixel each way: the percentiles of 170k pixels and of 670k
    # agree to well under the frame-to-frame scatter, at a quarter of the sort
    a = a[::2, ::2]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [None] * len(STATS)
    p = np.percentile(a, (50, 90, 95, 99))
    return [round(float(a.mean()), 4)] + [round(float(x), 4) for x in p]


def parse_roi(spec: str):
    y, x = spec.split()
    y0, y1 = (int(v) for v in y[1:].split("-"))
    x0, x1 = (int(v) for v in x[1:].split("-"))
    return slice(y0, y1), slice(x0, x1)


def _init(calib_path, ref, z, roi, gains):
    _CTX.update(calib=S.load_calibration(Path(calib_path)), ref=ref, z=z,
                roi=parse_roi(roi), gains=gains, maps={})


def _maps(cam, shape):
    m = _CTX["maps"]
    if cam not in m:
        m[cam] = (None if cam == _CTX["ref"] else
                  S.sample_map(_CTX["calib"], cam, _CTX["ref"], _CTX["z"], shape))
    return m[cam]


def measure_burst(job):
    burst_dir, frames = job
    _, cams = S.discover_burst(Path(burst_dir))
    if len(cams) < S.N_CAM:
        return {"burst": Path(burst_dir).name, "skipped": f"{len(cams)} cameras"}
    roi, gains = _CTX["roi"], _CTX["gains"]
    bands = {b: [i for i in range(S.N_CAM) if S.cam_band(i) == b] for b in S.BANDS}
    out = {"burst": Path(burst_dir).name, "frames": {}}
    stamp = None
    for f in frames:
        rec = {"raw": {}, "warp": {}, "band": {}}
        warped = {}
        for i in range(S.N_CAM):
            img, n, meta = S.load_camera_frame(*cams[i], f)
            if stamp is None and i == _CTX["ref"]:
                stamp = meta.get("timestamp")
            rec["raw"][S.cam_name(i)] = stats(img[roi])
            mp = _maps(i, img.shape)
            w = img if mp is None else S.remap_image(img, *mp)
            warped[i] = w[roi]
            rec["warp"][S.cam_name(i)] = stats(warped[i])
        if gains:
            for b, ids in bands.items():
                stack = np.stack([warped[i] / gains[S.cam_name(i)] for i in ids])
                rec["band"][str(b)] = stats(np.nanmean(stack, axis=0))
        out["frames"][str(f)] = rec
    out["timestamp"] = stamp
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="packed session dir holding burstNNNN/")
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="json lines, appended to")
    ap.add_argument("--ref", type=int, default=5, help="common view (default e05)")
    ap.add_argument("--z", type=float, default=550.0, help="plane the views are warped to, mm")
    ap.add_argument("--roi", default="y200-900 x250-1200",
                    help="region in the reference view (default the earlier analysis')")
    ap.add_argument("--every", type=int, default=1, help="use every Nth burst")
    ap.add_argument("--frames", default="0:90:3", help="start:stop:step within a burst")
    ap.add_argument("--bursts", default=None, help="first:last burst index, inclusive")
    ap.add_argument("--gain-bursts", type=int, default=12,
                    help="bursts the per-camera gains are measured on when --gains is absent")
    ap.add_argument("--gains", type=Path, default=None, help="json {camera: gain}")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    a, b, c = (int(v) for v in args.frames.split(":"))
    frames = list(range(a, b, c))
    dirs = sorted(p for p in args.session.iterdir() if p.is_dir() and p.name.startswith("burst"))
    if args.bursts:
        lo, hi = (int(v) for v in args.bursts.split(":"))
        dirs = [d for d in dirs if lo <= int(d.name[5:]) <= hi]
    dirs = dirs[::args.every]
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            done.add(json.loads(line)["burst"])
    todo = [d for d in dirs if d.name not in done]
    print(f"session : {args.session}  {len(dirs)} bursts selected, {len(todo)} to do, "
          f"frames {frames[0]}..{frames[-1]} step {c} ({len(frames)})", flush=True)

    gpath = args.out.with_suffix(".gains.json")
    if args.gains:
        gains = json.loads(args.gains.read_text())
    elif gpath.exists():
        gains = json.loads(gpath.read_text())
    else:
        # the gain of a view is its warped p50 over the band's median p50,
        # taken on a few bursts spread over the run and kept fixed afterwards
        probe = dirs[:: max(1, len(dirs) // args.gain_bursts)][:args.gain_bursts]
        _init(args.calib, args.ref, args.z, args.roi, None)
        ratios = {S.cam_name(i): [] for i in range(S.N_CAM)}
        for d in probe:
            r = measure_burst((d, frames[:: max(1, len(frames) // 5)]))
            for rec in r.get("frames", {}).values():
                for bnd in S.BANDS:
                    ids = [S.cam_name(i) for i in range(S.N_CAM) if S.cam_band(i) == bnd]
                    med = np.median([rec["warp"][k][1] for k in ids])
                    for k in ids:
                        ratios[k].append(rec["warp"][k][1] / med)
        gains = {k: float(np.median(v)) for k, v in ratios.items()}
        gpath.write_text(json.dumps(gains, indent=1))
        print("gains   : " + "  ".join(f"{k} {v:.3f}" for k, v in gains.items()), flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.calib, args.ref, args.z, args.roi, gains)) as ex, \
            args.out.open("a") as fh:
        for n, res in enumerate(ex.map(measure_burst, [(d, frames) for d in todo]), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / n:.1f}s/burst  eta {el / n * (len(todo) - n) / 60:.0f} min",
                      flush=True)


if __name__ == "__main__":
    main()
