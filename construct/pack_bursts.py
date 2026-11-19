#!/usr/bin/env python3
"""
pack_bursts.py - group the per-camera burst files of a session into one unit
per trigger, so downstream code opens a burst instead of hunting for 16 files.

Layout produced (default --format dir):

    <out>/<session>/burst0001/e00.npz          hardlink to the collected file
                              e01.npz
                              ...
                              manifest.json    which cameras, from where, when
    <out>/<session>/pack_index.json            state for resuming

With --format npz the same burst becomes a single file instead:

    <out>/<session>/burst0001.npz              members e00_frames_raw, e00_timestamps,
                                               e01_frames_raw, ...  (np.load reads it)
    <out>/<session>/burst0001.manifest.json

WHAT COUNTS AS ONE BURST
    Every camera answers the same trigger, so burst index N is the same
    physical event on all of them - the index, not the file name's timestamp,
    is the key. Those timestamps come from each Pi's own clock and in this
    array they spread over ~140 s for one and the same burst, which is why
    analyze/temporal_align.py exists. The index is only unreliable if a camera
    missed a trigger, so the packer checks every camera for gaps in its burst
    numbering and in its inter-burst interval, and says so before packing.

WHICH DIRECTORY A CAMERA'S FILES COME FROM
    Each camera directory is named <camera>_<the Pi's start time>, and a
    session normally holds one per camera. A collection run pointed at the
    wrong session can leave a second one - 20260728_073439 ended up with
    e10_20260729_053202, the NEXT day's run - and because the burst index is
    the key, its burst0001 would silently replace the right one. So per camera
    only the directory whose start time is closest to the session's own is
    used, and only if it is within --max-skew of it; every other directory is
    reported and left out.

CAMERAS THAT HAVE NOT ARRIVED YET
    A camera is included when its file for that burst is fully transferred:
    every complete burst file in a session has exactly the same size, so a
    short file is one that rsync has not finished (or that was interrupted).
    Those are skipped, never packed half way, and reported.

RESUMING
    Re-run the same command after more data lands. Each burst records the set
    of cameras it was packed from and a signature of those files; a burst is
    left alone when nothing changed, and topped up when a camera appears or a
    file is replaced. --format dir adds the new links; --format npz appends
    the new members to the existing archive. Neither redoes finished work.

Usage:
    python3 construct/pack_bursts.py --check  collected_data/20260728_073439
    python3 construct/pack_bursts.py          collected_data/20260728_073439
    python3 construct/pack_bursts.py --format npz --bursts 1-50 <session>
    python3 construct/pack_bursts.py --require 16 <session>     # only full sets

No third-party dependency: this runs on machines that have no numpy.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BURST_RE = re.compile(r"^(e\d{2})_burst(\d+)_(\d{8}_\d{6})_(\d{3})\.npz$")
CAM_DIR_RE = re.compile(r"^(e\d{2})_(\d{8}_\d{6})$")
ALL_CAMS = [f"e{i:02d}" for i in range(16)]


# ---------------------------------------------------------------- scanning ---
def _stamp(text: str):
    try:
        return datetime.strptime(text, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def pick_camera_dirs(session: Path, max_skew: float):
    """-> (directories to scan, [(directory, reason)] left out).

    One directory per camera: the one whose start time is closest to the
    session's (the session name, or failing that the median of the cameras'),
    and within max_skew seconds of it. A directory that does not look like
    <camera>_<start> is scanned as before, since its time cannot be judged.
    """
    dirs = sorted(p for p in session.iterdir() if p.is_dir())
    named = defaultdict(list)
    use = []
    for d in dirs:
        m = CAM_DIR_RE.match(d.name)
        t = _stamp(m.group(2)) if m else None
        if t is None:
            use.append(d)
        else:
            named[m.group(1)].append((t, d))
    ref = _stamp(session.name)
    if ref is None:
        starts = sorted(t for v in named.values() for t, _ in v)
        ref = starts[len(starts) // 2] if starts else None

    left_out = []
    for cam in sorted(named):
        cands = sorted(named[cam], key=lambda td: abs((td[0] - ref).total_seconds()))
        best_t, best = cands[0]
        skew = abs((best_t - ref).total_seconds())
        if skew <= max_skew:
            use.append(best)
        else:
            left_out.append((best, f"starts {skew / 3600:.1f} h from the session"))
        for _, d in cands[1:]:
            left_out.append((d, f"second directory for {cam}; using {best.name}"
                                if skew <= max_skew else
                                f"starts too far from the session as well"))
    return sorted(use), left_out


def scan_session(session: Path, cameras=None, settle=60.0, max_skew=1800.0):
    """burst index -> {camera: (path, size, mtime_ns, timestamp)} for ready files.

    Also returns the files that exist but are not usable yet, so the caller can
    report them rather than silently pretending they are missing, and the
    camera directories that were left out (see pick_camera_dirs).
    """
    by_size = defaultdict(int)
    found = defaultdict(dict)
    cam_dirs, left_out = pick_camera_dirs(session, max_skew)
    for cam_dir in cam_dirs:
        for f in cam_dir.iterdir():
            m = BURST_RE.match(f.name)
            if not m:
                continue
            cam, idx, ts, ms = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            if cameras and cam not in cameras:
                continue
            st = f.stat()
            by_size[st.st_size] += 1
            found[idx][cam] = (f, st.st_size, st.st_mtime_ns, f"{ts}.{ms}")

    if not by_size:
        return {}, [], 0, left_out
    # A complete burst file is always the same size; anything else is in flight.
    canonical = max(by_size.items(), key=lambda kv: (kv[1], kv[0]))[0]

    now = time.time()
    ready, pending = defaultdict(dict), []
    for idx, cams in found.items():
        for cam, rec in cams.items():
            path, size, mtime_ns, ts = rec
            if size != canonical:
                pending.append((cam, idx, path, size, "incomplete"))
            elif settle > 0 and now - (mtime_ns / 1e9) < settle:
                pending.append((cam, idx, path, size, "just written"))
            else:
                ready[idx][cam] = rec
    return dict(ready), pending, canonical, left_out


def check_indexing(ready, pending=()):
    """Warn where a burst index may not mean the same event on every camera.

    A camera that missed a trigger keeps counting, so from that point on its
    burst N is somebody else's burst N+1. Two symptoms give that away: a hole
    in the numbering, and an inter-burst interval that jumps.
    """
    per_cam = defaultdict(dict)
    for idx, cams in ready.items():
        for cam, rec in cams.items():
            per_cam[cam][idx] = rec[3]

    # a hole where the file merely has not finished transferring is not a
    # missed trigger, and must not be reported as one
    not_yet = {(cam, idx) for cam, idx, *_ in pending}

    problems = []
    for cam in sorted(per_cam):
        idxs = sorted(per_cam[cam])
        holes = [i for i in range(idxs[0], idxs[-1] + 1)
                 if i not in per_cam[cam] and (cam, i) not in not_yet]
        waiting = [i for i in range(idxs[0], idxs[-1] + 1) if (cam, i) in not_yet]
        stamps = []
        for i in idxs:
            try:
                stamps.append((i, datetime.strptime(per_cam[cam][i][:15], "%Y%m%d_%H%M%S")))
            except ValueError:
                pass
        deltas = [(stamps[k][0], (stamps[k + 1][1] - stamps[k][1]).total_seconds())
                  for k in range(len(stamps) - 1)]
        if deltas:
            values = [round(d) for _, d in deltas]
            nominal = max(set(values), key=values.count)
            odd = [(i, d) for i, d in deltas if abs(d - nominal) > max(5.0, 0.1 * nominal)]
        else:
            nominal, odd = 0, []
        # an interval jump straight across a file that is still transferring
        # is explained by that file, not by a missed trigger
        odd = [(i, d) for i, d in odd if not any(i < w <= i + 2 for w in waiting)]
        if holes or odd:
            problems.append((cam, len(idxs), holes, nominal, odd, waiting))
    return per_cam, problems


# ----------------------------------------------------------------- packing ---
def burst_signature(cams):
    """Fingerprint of the sources a burst was packed from."""
    h = hashlib.sha1()
    for cam in sorted(cams):
        path, size, mtime_ns, _ = cams[cam]
        h.update(f"{cam}:{size}:{mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def link_file(src: Path, dst: Path, mode: str):
    """hard | sym | copy, with an automatic fall back when hardlinks cannot cross."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hard":
        try:
            os.link(src, dst)
            return "hard"
        except OSError:
            mode = "sym"
    if mode == "sym":
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return "sym"
    shutil.copy2(src, dst)
    return "copy"


def already_linked(dst: Path, src: Path, size: int) -> bool:
    """True when dst already stands for exactly this source file.

    Checked per link mode, because a symlink and a hardlink to the same file
    look nothing alike on disk and re-doing either one is pure waste.
    """
    if dst.is_symlink():
        try:
            return (dst.parent / os.readlink(dst)).resolve() == src.resolve()
        except OSError:
            return False
    if not dst.exists():
        return False
    st = dst.stat()
    return st.st_size == size and (st.st_ino == src.stat().st_ino
                                   or st.st_mtime == src.stat().st_mtime)


def pack_dir(out_dir: Path, idx: int, cams, link_mode: str, dry: bool):
    """One directory per burst, one entry per camera. Adds only what is missing."""
    bdir = out_dir / f"burst{idx:04d}"
    added = []
    if not dry:
        bdir.mkdir(parents=True, exist_ok=True)
    for cam in sorted(cams):
        dst = bdir / f"{cam}.npz"
        src = cams[cam][0]
        if already_linked(dst, src, cams[cam][1]):
            continue          # this camera is already in place
        if not dry:
            link_file(src, dst, link_mode)
        added.append(cam)
    return bdir, added


def pack_npz(out_dir: Path, idx: int, cams, dry: bool):
    """One npz per burst; members are prefixed with the camera name.

    Members are copied as bytes - np.savez stores them uncompressed - so this
    needs no numpy and does no re-encoding. New cameras are appended to an
    existing archive rather than rebuilding it.
    """
    dst = out_dir / f"burst{idx:04d}.npz"
    added = []
    if dry:
        return dst, sorted(cams)
    existing = set()
    if dst.exists():
        with zipfile.ZipFile(dst) as z:
            existing = set(z.namelist())
    mode = "a" if dst.exists() else "w"
    tmp = dst.with_suffix(".npz.part")
    if mode == "w" and tmp.exists():
        tmp.unlink()
    target = dst if mode == "a" else tmp
    with zipfile.ZipFile(target, mode, zipfile.ZIP_STORED, allowZip64=True) as zout:
        for cam in sorted(cams):
            src = cams[cam][0]
            wrote = False
            with zipfile.ZipFile(src) as zin:
                for name in zin.namelist():
                    member = f"{cam}_{name}"
                    if member in existing:
                        continue
                    with zin.open(name) as fsrc, zout.open(member, "w") as fdst:
                        shutil.copyfileobj(fsrc, fdst, 4 << 20)
                    existing.add(member)
                    wrote = True
            if wrote:                      # a camera already in the archive is left alone
                added.append(cam)
    if mode == "w":
        tmp.replace(dst)
    return dst, added


def write_manifest(path: Path, idx: int, cams, missing, canonical, dry):
    stamps = {c: cams[c][3] for c in sorted(cams)}
    spread = None
    try:
        ts = [datetime.strptime(s[:15], "%Y%m%d_%H%M%S") for s in stamps.values()]
        spread = (max(ts) - min(ts)).total_seconds()
    except ValueError:
        pass
    doc = {
        "burst": idx,
        "cameras": sorted(cams),
        "missing": sorted(missing),
        "n_cameras": len(cams),
        "file_timestamps": stamps,
        "timestamp_spread_s": spread,
        "note": ("file timestamps come from each Pi's own clock; use "
                 "analyze/temporal_align.py for frame-level alignment"),
        "sources": {c: str(cams[c][0]) for c in sorted(cams)},
        "source_size": canonical,
        "packed_at": datetime.now().isoformat(timespec="seconds"),
    }
    if not dry:
        path.write_text(json.dumps(doc, indent=2))
    return doc


# --------------------------------------------------------------------- cli ---
def parse_bursts(spec):
    if not spec:
        return None
    want = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            want.update(range(int(a), int(b) + 1))
        elif part:
            want.add(int(part))
    return want


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path,
                    help="collected session directory, e.g. collected_data/20260728_073439")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default: <session>/../packed)")
    ap.add_argument("--format", choices=("dir", "npz"), default="dir",
                    help="dir: burst0001/e00.npz per camera (default, no extra disk "
                         "space with hardlinks) | npz: one merged file per burst")
    ap.add_argument("--link", choices=("hard", "sym", "copy"), default="hard",
                    help="how --format dir materialises entries (default hard)")
    ap.add_argument("--bursts", default=None, help="subset, e.g. 1-100,205")
    ap.add_argument("--cameras", default=None, help="subset, e.g. e00,e01,e04")
    ap.add_argument("--require", type=int, default=1,
                    help="skip bursts with fewer cameras than this (default 1)")
    ap.add_argument("--settle", type=float, default=60.0,
                    help="ignore files written within the last N seconds (default 60)")
    ap.add_argument("--max-skew", type=float, default=1800.0,
                    help="seconds a camera directory's start may differ from the "
                         "session's before it is left out (default 1800)")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true", help="re-pack even when unchanged")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    session = args.session.resolve()
    if not session.is_dir():
        sys.exit(f"no such session directory: {session}")
    cameras = set(args.cameras.split(",")) if args.cameras else None
    want = parse_bursts(args.bursts)

    out_root = args.out or session.parent / "packed"
    out_dir = Path(out_root).resolve() / session.name

    print(f"session : {session}")
    print(f"output  : {out_dir}   format={args.format}"
          + (f" link={args.link}" if args.format == "dir" else ""))

    ready, pending, canonical, left_out = scan_session(session, cameras, args.settle,
                                                       args.max_skew)
    for d, why in left_out:
        print(f"LEFT OUT: {d.name}  ({why})")
    if not ready:
        sys.exit("no complete burst files found")
    print(f"burst files: {sum(len(c) for c in ready.values())} usable, "
          f"{len(pending)} not ready, {human(canonical)} each")

    # ---- coverage, and whether the index means the same event everywhere ----
    per_cam, problems = check_indexing(ready, pending)
    print(f"cameras : {len(per_cam)} ({', '.join(sorted(per_cam))})")
    runs = []
    for idx in sorted(ready):
        n = len(ready[idx])
        if runs and runs[-1][2] == n and runs[-1][1] == idx - 1:
            runs[-1][1] = idx
        else:
            runs.append([idx, idx, n])
    print("coverage:")
    for a, b, n in runs:
        miss = sorted(set(cameras or ALL_CAMS) - set(ready[a]))
        print(f"  burst {a:4d}-{b:4d}  {n:2d} cam  ({b - a + 1:4d} bursts)"
              f"   missing: {','.join(miss) if miss else '-'}")
    if pending:
        print(f"not ready yet ({len(pending)}), left for a later run:")
        for cam, idx, path, size, why in sorted(pending)[:12]:
            print(f"  {cam} burst{idx:04d}  {human(size)} of {human(canonical)}  ({why})")
        if len(pending) > 12:
            print(f"  ... and {len(pending) - 12} more")
    if problems:
        print("burst numbering to check before trusting the pairing:")
        for cam, n, holes, nominal, odd, waiting in problems:
            if holes:
                print(f"  {cam}: {len(holes)} missing index/es, e.g. {holes[:5]}")
            if odd:
                print(f"  {cam}: interval jumps from {nominal:.0f}s at burst {[i for i, _ in odd][:5]}"
                      f" -> a missed trigger would shift every later index")
    else:
        print("burst numbering: continuous and evenly spaced on every camera"
              + (" (bar the files still in transfer)" if pending else ""))

    # ------------------------------ state ---------------------------------
    index_path = out_dir / "pack_index.json"
    state = {"session": session.name, "format": args.format, "bursts": {}}
    if index_path.exists():
        try:
            state = json.loads(index_path.read_text())
            if state.get("format") != args.format:
                print(f"note: existing index was packed as '{state.get('format')}'; "
                      f"writing '{args.format}' alongside it")
                state["format"] = args.format
        except json.JSONDecodeError:
            print("note: unreadable pack_index.json, starting a fresh one")
            state = {"session": session.name, "format": args.format, "bursts": {}}
    known = state.setdefault("bursts", {})

    todo, skipped, thin = [], 0, 0
    for idx in sorted(ready):
        if want and idx not in want:
            continue
        cams = ready[idx]
        if len(cams) < args.require:
            thin += 1
            continue
        sig = burst_signature(cams)
        rec = known.get(str(idx))
        if rec and rec.get("sig") == sig and not args.force:
            skipped += 1
            continue
        todo.append((idx, cams, sig, rec))

    print(f"\nto pack : {len(todo)} burst(s)"
          + (f", {skipped} already current" if skipped else "")
          + (f", {thin} below --require {args.require}" if thin else ""))
    if args.check:
        for idx, cams, _, rec in todo[:10]:
            grew = f" (was {len(rec['cameras'])} cam)" if rec else ""
            print(f"  burst{idx:04d}: {len(cams)} cam{grew}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        print("--check: nothing was written")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    packed_bytes = 0
    for n, (idx, cams, sig, rec) in enumerate(todo, 1):
        missing = sorted(set(cameras or ALL_CAMS) - set(cams))
        if args.format == "dir":
            target, added = pack_dir(out_dir, idx, cams, args.link, args.dry_run)
            manifest = target / "manifest.json"
        else:
            target, added = pack_npz(out_dir, idx, cams, args.dry_run)
            manifest = target.with_suffix(".manifest.json")
        write_manifest(manifest, idx, cams, missing, canonical, args.dry_run)
        if args.format == "npz" or args.link == "copy":
            packed_bytes += len(added) * canonical

        known[str(idx)] = {
            "sig": sig,
            "cameras": sorted(cams),
            "missing": missing,
            "path": target.name,
            "packed_at": datetime.now().isoformat(timespec="seconds"),
        }
        grew = f" (+{','.join(added)})" if added and rec else ""
        if n <= 5 or n % 25 == 0 or n == len(todo):
            rate = (f"  {human(packed_bytes / max(time.time() - t0, 1e-9))}/s"
                    if packed_bytes else "")
            print(f"  [{n}/{len(todo)}] burst{idx:04d}: {len(cams)} cam{grew}{rate}")
        if not args.dry_run and n % 50 == 0:
            state["updated"] = datetime.now().isoformat(timespec="seconds")
            index_path.write_text(json.dumps(state, indent=2))

    if not args.dry_run:
        state["updated"] = datetime.now().isoformat(timespec="seconds")
        state["canonical_size"] = canonical
        index_path.write_text(json.dumps(state, indent=2))

    dt = time.time() - t0
    print(f"\npacked {len(todo)} burst(s) in {dt:.1f}s"
          + (f", {human(packed_bytes)} written" if packed_bytes else " (no data copied)"))
    print(f"index  : {index_path}")
    print("re-run the same command after more cameras upload; finished bursts are kept "
          "and the new cameras are added to them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
