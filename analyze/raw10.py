#!/usr/bin/env python3
"""
raw10.py - Utilities for 10-bit CSI2-packed raw burst npz files.

Burst npz layout (CAPTURE_RAW=True in camera_agent.py):
    frames_raw    : (N, H, stride) uint8, CSI2-packed RAW10
                    (4 pixels in 5 bytes: 4x MSB bytes + 1 LSB byte)
    timestamps    : (N,) str "%Y%m%d_%H%M%S_%f" (ms)
    raw_format    : str, e.g. "R10_CSI2P"
    raw_shape     : (2,) [H, W]
    black_levels  : (4,) sensor black levels on a 16-bit scale (>>6 = 10bit)
    analogue_gain : float
    mono_s        : (N,) float64, time.monotonic() per frame (since 2026-11-25)
    boot_id       : str, /proc/sys/kernel/random/boot_id of the Pi's boot
    The wall-clock "timestamps" can step back after a power cut (no RTC);
    within one boot_id, mono_s orders frames and bursts exactly.

Typical use in analysis:
    from raw10 import load_burst
    frames, ts, meta = load_burst("e00_burst0002_....npz")
    # frames: (N, 1088, 1456) uint16, linear, black level subtracted
"""

import numpy as np


def unpack_raw10(packed: np.ndarray, width: int) -> np.ndarray:
    """Unpack CSI2 RAW10 (4 pixels / 5 bytes) along the last axis.

    Args:
        packed: (..., stride) uint8 array (stride >= width*5/4)
        width:  number of pixels per row

    Returns:
        (..., width) uint16 array with 10-bit values
    """
    nbytes = (width + 3) // 4 * 5
    b = packed[..., :nbytes].astype(np.uint16)
    b0 = b[..., 0::5]
    b1 = b[..., 1::5]
    b2 = b[..., 2::5]
    b3 = b[..., 3::5]
    b4 = b[..., 4::5]
    p0 = (b0 << 2) | (b4 & 0x03)
    p1 = (b1 << 2) | ((b4 >> 2) & 0x03)
    p2 = (b2 << 2) | ((b4 >> 4) & 0x03)
    p3 = (b3 << 2) | ((b4 >> 6) & 0x03)
    out = np.stack([p0, p1, p2, p3], axis=-1)
    out = out.reshape(*packed.shape[:-1], -1)
    return out[..., :width]


def load_burst(path, subtract_black: bool = True):
    """Load one burst npz (raw or 8-bit) as an (N, H, W) array.

    Returns:
        frames: (N, H, W) uint16 (raw, linear) or uint8 (legacy Y8)
        timestamps: (N,) str array
        meta: dict with bit_depth, black_level (10-bit units),
              analogue_gain, raw_format
    """
    d = np.load(path)
    ts = d["timestamps"]

    if "frames_raw" in d.files:
        H, W = (int(x) for x in d["raw_shape"])
        frames = np.stack([unpack_raw10(f, W) for f in d["frames_raw"]])

        bl = d["black_levels"] if "black_levels" in d.files else np.array([])
        black10 = int(bl.flat[0]) >> 6 if bl.size else 0
        if subtract_black and black10 > 0:
            frames = np.maximum(frames, black10) - black10

        gain = float(d["analogue_gain"]) if "analogue_gain" in d.files else 0.0
        meta = {
            "bit_depth": 10,
            "black_level": black10,
            "black_subtracted": bool(subtract_black and black10 > 0),
            "analogue_gain": gain,
            "raw_format": str(d["raw_format"]) if "raw_format" in d.files else "",
            # absent in files written before 2026-11-25
            "mono_s": d["mono_s"] if "mono_s" in d.files else None,
            "boot_id": str(d["boot_id"]) if "boot_id" in d.files else "",
        }
        return frames, ts, meta

    # Legacy 8-bit Y frames
    return d["frames"], ts, {
        "bit_depth": 8, "black_level": 0,
        "black_subtracted": False, "analogue_gain": 0.0, "raw_format": "",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("Usage: python raw10.py <burst.npz>")
    frames, ts, meta = load_burst(sys.argv[1])
    print(f"frames: {frames.shape} {frames.dtype}")
    print(f"meta:   {meta}")
    print(f"stats:  mean={frames.mean():.1f} min={frames.min()} "
          f"max={frames.max()}")
