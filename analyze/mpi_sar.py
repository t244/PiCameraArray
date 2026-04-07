#!/usr/bin/env python3
"""
MPI-based Synthetic Aperture Rendering for foreground removal.

Builds a Multiplane Image (MPI) representation via plane-sweep stereo,
estimates per-plane alpha opacity, zeros out near-depth alpha planes to
remove foreground occluders (e.g. fences), and re-composites a clean
background image.

Algorithm overview:
  1. Load & undistort 16 camera images
  2. Compute plane-sweep stereo cost volume (at 0.25x resolution)
  3. Convert cost to per-plane alpha via softmax
  4. Remove foreground by zeroing near-depth planes
  5. Composite MPI back-to-front to produce clean background

Usage:
    python analyze/mpi_sar.py <image_dir> --calib calibration_results.npz
    python analyze/mpi_sar.py <image_dir> --calib calibration_results.npz --method median
    python analyze/mpi_sar.py <image_dir> --calib calibration_results.npz --method mpi --soft-removal
    python analyze/mpi_sar.py <image_dir> --calib calibration_results.npz --method defencing --composite ransac

Based on the MPI synthetic aperture survey in mpi_sar/.
"""

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════════

def load_calibration(npz_path: str) -> dict:
    """Load calibration data from .npz file.

    Supports two formats:
      - Array format: camera_matrices (16,3,3), dist_coeffs (16,5),
        R_mats (16,3,3), tvecs (16,3), image_size (2,)
      - Per-camera format: e00_camera_matrix, e00_board_rvec, etc.

    Returns:
        dict with keys: K (16,3,3), dist (16,5), R_mats (16,3,3),
        tvecs (16,3), image_size (2,)
    """
    data = np.load(npz_path, allow_pickle=True)

    # Try array format first
    if "camera_matrices" in data:
        K = data["camera_matrices"].astype(np.float64)
        dist = data["dist_coeffs"].astype(np.float64)
        tvecs = data["tvecs"].astype(np.float64)
        image_size = data["image_size"]

        if "R_mats" in data:
            R_mats = data["R_mats"].astype(np.float64)
        elif "rvecs" in data:
            R_mats = build_rotation_matrices(data["rvecs"].astype(np.float64))
        else:
            raise KeyError("calibration needs R_mats or rvecs")

        return dict(K=K, dist=dist, R_mats=R_mats, tvecs=tvecs,
                    image_size=image_size)

    # Per-camera format (e00_camera_matrix, ...)
    n_cameras = 16
    K = np.zeros((n_cameras, 3, 3), dtype=np.float64)
    dist_list = [None] * n_cameras
    R_mats = np.zeros((n_cameras, 3, 3), dtype=np.float64)
    tvecs = np.zeros((n_cameras, 3), dtype=np.float64)

    # Read image_size from calibration (per-camera key), fallback to default
    image_size = np.array([1456, 1088])
    if "e00_image_size" in data:
        image_size = np.array(data["e00_image_size"]).flatten().astype(int)

    for i in range(n_cameras):
        prefix = f"e{i:02d}"

        km_key = f"{prefix}_camera_matrix"
        if km_key in data:
            K[i] = data[km_key].astype(np.float64)

        dc_key = f"{prefix}_dist_coeffs"
        if dc_key in data:
            # Keep full distortion vector (may be 5, 8, 12, or 14 coefficients)
            dist_list[i] = data[dc_key].astype(np.float64).reshape(-1)

        # Board poses -> rotation matrix
        rv_key = f"{prefix}_board_rvec"
        tv_key = f"{prefix}_board_tvec"
        if rv_key in data and tv_key in data:
            rvec = data[rv_key].astype(np.float64).reshape(3)
            R_mats[i], _ = cv2.Rodrigues(rvec)
            # Calibration tvecs are in meters — convert to mm
            tvecs[i] = data[tv_key].astype(np.float64).reshape(3) * 1000.0

        rm_key = f"{prefix}_board_rotation_matrix"
        if rm_key in data:
            R_mats[i] = data[rm_key].astype(np.float64)

    # Stack dist_coeffs: use max length, zero-pad shorter ones
    max_len = max(len(d) for d in dist_list if d is not None)
    dist = np.zeros((n_cameras, max_len), dtype=np.float64)
    for i in range(n_cameras):
        if dist_list[i] is not None:
            dist[i, :len(dist_list[i])] = dist_list[i]

    return dict(K=K, dist=dist, R_mats=R_mats, tvecs=tvecs,
                image_size=image_size)


def build_rotation_matrices(rvecs: np.ndarray) -> np.ndarray:
    """Convert (N, 3) Rodrigues vectors to (N, 3, 3) rotation matrices."""
    N = rvecs.shape[0]
    R_mats = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(N):
        R_mats[i], _ = cv2.Rodrigues(rvecs[i])
    return R_mats


def compute_relative_poses(
    R_mats: np.ndarray, tvecs: np.ndarray, ref_idx: int = 0
) -> tuple:
    """Compute poses relative to the reference camera.

    Args:
        R_mats: (N, 3, 3) world-to-camera rotation matrices.
        tvecs:  (N, 3) translation vectors.
        ref_idx: Reference camera index.

    Returns:
        (R_rel, t_rel): (N,3,3) and (N,3) relative transforms.
    """
    N = R_mats.shape[0]
    R_ref = R_mats[ref_idx]
    t_ref = tvecs[ref_idx]

    R_rel = np.zeros_like(R_mats)
    t_rel = np.zeros_like(tvecs)

    for i in range(N):
        R_rel[i] = R_mats[i] @ R_ref.T
        t_rel[i] = tvecs[i] - R_rel[i] @ t_ref

    return R_rel, t_rel


def compute_depth_planes(z_near: float, z_far: float, D: int) -> np.ndarray:
    """Generate D depth values linearly spaced in disparity (1/z).

    Plane 0 = farthest (z_far), plane D-1 = nearest (z_near).

    Returns:
        (D,) float32 depth values in mm.
    """
    disparities = np.linspace(1.0 / z_far, 1.0 / z_near, D, dtype=np.float32)
    return (1.0 / disparities).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Image I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_images(directory: Path, pattern: str = "e*.png") -> tuple:
    """Load camera images as (N, H, W) float32 in [0, 1].

    Searches for images matching the pattern directly in the directory.
    Falls back to TIFF if no PNG found. Also handles per-camera
    subdirectory layouts (e.g. collected_data/) by taking the first
    image from each e{NN}_* subdirectory.

    Returns:
        (images, cam_indices): images is (N, H, W) float32 array,
        cam_indices is list of int camera indices parsed from filenames
        (e.g. [0, 1, 2, ..., 12, 14, 15] if e13 is missing).
    """
    paths = sorted(directory.glob(pattern))
    if not paths:
        paths = sorted(directory.glob("e*.tif"))

    # Fallback: per-camera subdirectories (e00_*/e00_*.png, ...)
    if not paths:
        subdirs = sorted(directory.glob("e[0-9][0-9]_*"))
        for sd in subdirs:
            if sd.is_dir():
                candidates = sorted(sd.glob("*.png")) + sorted(sd.glob("*.tif"))
                if candidates:
                    paths.append(candidates[0])

    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")

    images = []
    cam_indices = []
    for p in paths:
        # Parse camera index from filename (e.g. e00.png -> 0, e14.png -> 14)
        m = re.match(r"e(\d{2})", p.stem)
        cam_idx = int(m.group(1)) if m else len(images)

        if p.suffix.lower() in (".tif", ".tiff"):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img.astype(np.float32) / 1023.0)
                cam_indices.append(cam_idx)
        else:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.astype(np.float32) / 255.0)
                cam_indices.append(cam_idx)

    print(f"Loaded {len(images)} images from {directory} "
          f"(cameras: {cam_indices})")
    return np.stack(images, axis=0), cam_indices


def undistort_images(
    images: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """Undistort all N images. K and dist are (N, ...) arrays."""
    N = images.shape[0]
    out = np.empty_like(images)
    for i in range(N):
        h, w = images[i].shape
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K[i], dist[i], (w, h), 0, (w, h)
        )
        out[i] = cv2.undistort(images[i], K[i], dist[i], None, new_K)
    return out


def downsample(images: np.ndarray, scale: float) -> np.ndarray:
    """Downsample (N, H, W) images by scale factor."""
    N, H, W = images.shape
    new_h = int(H * scale)
    new_w = int(W * scale)
    out = np.empty((N, new_h, new_w), dtype=np.float32)
    for i in range(N):
        out[i] = cv2.resize(images[i], (new_w, new_h),
                            interpolation=cv2.INTER_AREA)
    return out


def save_image(image: np.ndarray, path: Path):
    """Save a float32 [0,1] image as uint8 PNG."""
    out = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


# ═══════════════════════════════════════════════════════════════════════════════
# Homography & Warping
# ═══════════════════════════════════════════════════════════════════════════════

def compute_homography(
    K_ref: np.ndarray, K_i: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray, z: float
) -> np.ndarray:
    """Compute plane-induced homography for fronto-parallel plane at depth z.

    Maps source camera pixels to reference camera pixels (for warpPerspective).
    R_rel, t_rel describe the ref→source transform, so we invert to get
    source→ref: R_i2r = R_rel^T, t_i2r = -R_rel^T @ t_rel.

    H = K_ref @ (R_i2r + t_i2r @ n^T / z) @ K_i^{-1}

    Args:
        K_ref: (3, 3) reference camera intrinsics.
        K_i:   (3, 3) source camera intrinsics.
        R_rel: (3, 3) relative rotation (ref → source camera).
        t_rel: (3,) relative translation (ref → source camera, mm).
        z:     Depth of the virtual plane (mm), must be > 0.

    Returns:
        (3, 3) float64 homography matrix.
    """
    if z <= 0:
        raise ValueError(f"Depth z must be > 0, got {z}")

    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # Invert relative pose: ref→i  =>  i→ref
    R_i2r = R_rel.T
    t_i2r = -R_rel.T @ t_rel
    H = K_ref @ (R_i2r + np.outer(t_i2r, n) / z) @ np.linalg.inv(K_i)
    return H.astype(np.float64)


def compute_all_homographies(
    K: np.ndarray, R_rel: np.ndarray, t_rel: np.ndarray,
    z: float, ref_idx: int = 0
) -> np.ndarray:
    """Compute homographies from all N cameras to reference at depth z.

    Returns:
        (N, 3, 3) float64 homography matrices. H[ref_idx] = identity.
    """
    N = K.shape[0]
    H_all = np.zeros((N, 3, 3), dtype=np.float64)
    K_ref = K[ref_idx]

    for i in range(N):
        if i == ref_idx:
            H_all[i] = np.eye(3, dtype=np.float64)
        else:
            H_all[i] = compute_homography(K_ref, K[i], R_rel[i], t_rel[i], z)

    return H_all


def warp_image(
    image: np.ndarray, H: np.ndarray, output_size: tuple
) -> np.ndarray:
    """Warp a float32 image using homography H.

    Args:
        image: (H, W) float32 source image.
        H: (3, 3) homography.
        output_size: (width, height).

    Returns:
        Warped (height, width) float32 image.
    """
    return cv2.warpPerspective(
        image, H, output_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )


def compute_validity_mask(
    H: np.ndarray, src_shape: tuple, output_size: tuple
) -> np.ndarray:
    """Binary validity mask: 1 where warped pixels are in-bounds."""
    ones = np.ones(src_shape, dtype=np.float32)
    mask = cv2.warpPerspective(
        ones, H, output_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )
    return (mask > 0.5).astype(np.float32)


def batch_warp_threaded(
    images: np.ndarray, homographies: np.ndarray,
    output_size: tuple, max_workers: int = 4
) -> tuple:
    """Warp N images in parallel. Returns (warped_list, mask_list)."""

    def _warp_one(idx):
        warped = warp_image(images[idx], homographies[idx], output_size)
        mask = compute_validity_mask(
            homographies[idx], images[idx].shape, output_size
        )
        return warped, mask

    N = images.shape[0]
    warped_list = [None] * N
    mask_list = [None] * N

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_warp_one, i): i for i in range(N)}
        for fut in futures:
            i = futures[fut]
            warped_list[i], mask_list[i] = fut.result()

    return warped_list, mask_list


# ═══════════════════════════════════════════════════════════════════════════════
# Defencing v2 — Dense Defocused Mesh Removal
# ═══════════════════════════════════════════════════════════════════════════════

def compute_overlap_crop(
    masks: np.ndarray, margin: int = 10
) -> tuple:
    """Rectangular region where ALL views have valid data.

    Args:
        masks: (N, H, W) float32 validity masks.
        margin: Shrink crop by this many pixels on each side.

    Returns:
        (y0, y1, x0, x1) crop coordinates.
    """
    all_valid = np.all(masks > 0.5, axis=0)  # (H, W)
    rows = np.where(all_valid.any(axis=1))[0]
    cols = np.where(all_valid.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        H, W = masks.shape[1], masks.shape[2]
        return (0, H, 0, W)
    y0, y1 = int(rows[0]) + margin, int(rows[-1]) + 1 - margin
    x0, x1 = int(cols[0]) + margin, int(cols[-1]) + 1 - margin
    y0, x0 = max(y0, 0), max(x0, 0)
    y1 = max(y1, y0 + 1)
    x1 = max(x1, x0 + 1)
    return (y0, y1, x0, x1)


def _local_sharpness_weight(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Per-pixel sharpness as mesh indicator.

    Background is in-focus (high gradient energy), mesh is defocused (smooth).
    Returns (H, W) float32 in [0, 1].
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_32F)
    energy = lap * lap
    # Box filter over ~2*sigma window to get local energy density
    ksize = max(3, int(4 * sigma) | 1)
    energy = cv2.blur(energy, (ksize, ksize))
    # Normalize to [0, 1]
    p99 = np.percentile(energy, 99)
    if p99 < 1e-10:
        return np.ones_like(image)
    w = np.clip(energy / p99, 0.0, 1.0)
    return w.astype(np.float32)


def _cross_view_consistency_weight(
    warped_images: np.ndarray, masks: np.ndarray,
    sigma: float = 0.03
) -> np.ndarray:
    """Per-view consistency with robust consensus.

    Consensus = trimmed mean of views in percentile [60, 90] to avoid
    dark-mesh and bright-specular outliers.
    Returns (N, H, W) float32 in [0, 1].
    """
    N, H, W = warped_images.shape
    masked = warped_images.copy()
    masked[masks < 0.5] = np.nan

    with np.errstate(all='ignore'):
        p60 = np.nanpercentile(masked, 60, axis=0)
        p90 = np.nanpercentile(masked, 90, axis=0)
    # Trimmed mean: mean of values in [p60, p90]
    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    for i in range(N):
        valid = (masks[i] > 0.5)
        in_range = valid & (warped_images[i] >= p60) & (warped_images[i] <= p90)
        acc += np.where(in_range, warped_images[i], 0.0)
        cnt += in_range.astype(np.float64)
    consensus = np.where(cnt > 0, acc / cnt, np.nanmedian(masked, axis=0))
    consensus = np.nan_to_num(consensus, nan=0.0).astype(np.float32)

    weights = np.zeros_like(warped_images)
    for i in range(N):
        diff = warped_images[i] - consensus
        w = np.exp(-0.5 * (diff / sigma) ** 2)
        w[masks[i] < 0.5] = 0.0
        weights[i] = w.astype(np.float32)
    return weights


def _percentile_deviation_weight(
    warped_images: np.ndarray, masks: np.ndarray,
    high_pct: float = 95.0
) -> np.ndarray:
    """Deviation from the 'clean ceiling' brightness estimate.

    Views below the ceiling are mesh-attenuated; views above are specular.
    Returns (N, H, W) float32 in [0, 1].
    """
    N = warped_images.shape[0]
    masked = warped_images.copy()
    masked[masks < 0.5] = np.nan
    with np.errstate(all='ignore'):
        ceiling = np.nanpercentile(masked, high_pct, axis=0).astype(np.float32)
    ceiling = np.nan_to_num(ceiling, nan=1.0)
    ceiling = np.maximum(ceiling, 1e-6)

    weights = np.zeros_like(warped_images)
    for i in range(N):
        ratio = warped_images[i] / ceiling
        # Dark mesh: penalize quadratically
        w_dark = np.clip(ratio, 0.0, 1.0) ** 2
        # Specular bright: also penalize
        w_spec = np.where(ratio > 1.05, np.clip(2.0 - ratio, 0.0, 1.0), 1.0)
        w = w_dark * w_spec
        w[masks[i] < 0.5] = 0.0
        weights[i] = w.astype(np.float32)
    return weights


def estimate_mesh_weights(
    warped_images: np.ndarray, masks: np.ndarray,
    sharpness_sigma: float = 3.0,
    consistency_sigma: float = 0.03,
    percentile_ref: float = 95.0,
    smooth_sigma: float = 5.0
) -> np.ndarray:
    """Per-view, per-pixel mesh detection weights.

    Combines three cues:
      1. Local sharpness (background in-focus vs mesh defocused)
      2. Cross-view consistency (mesh shifts between views)
      3. Percentile deviation (mesh attenuates or reflects)

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks.
        sharpness_sigma: Gaussian scale for sharpness measurement.
        consistency_sigma: Bandwidth for consistency weighting.
        percentile_ref: Reference percentile for ceiling estimate.
        smooth_sigma: Final smoothing of weight maps.

    Returns:
        (N, H, W) float32 weights in [0, 1].
    """
    N = warped_images.shape[0]

    # Cue 1: per-view sharpness
    sharpness = np.zeros_like(warped_images)
    for i in range(N):
        sharpness[i] = _local_sharpness_weight(warped_images[i], sharpness_sigma)

    # Cue 2: cross-view consistency
    consistency = _cross_view_consistency_weight(
        warped_images, masks, consistency_sigma
    )

    # Cue 3: percentile deviation
    pct_dev = _percentile_deviation_weight(
        warped_images, masks, percentile_ref
    )

    # Combine: product of three cues
    weights = sharpness * consistency * pct_dev

    # Smooth to avoid hard edges
    if smooth_sigma > 0:
        for i in range(N):
            weights[i] = cv2.GaussianBlur(weights[i], (0, 0), smooth_sigma)

    # Ensure at least one view has nonzero weight per pixel
    wsum = weights.sum(axis=0)
    dead = wsum < 1e-8
    if dead.any():
        for i in range(N):
            weights[i][dead] = masks[i][dead]

    return weights.astype(np.float32)


def _refine_tile_depth(
    ref_tile: np.ndarray,
    warped_tiles: np.ndarray,
    weight_tiles: np.ndarray,
    mask_tiles: np.ndarray,
    depth_candidates: np.ndarray,
    K_ref: np.ndarray, K_all: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray,
    ref_idx: int,
    tile_origin: tuple,
    base_depth: float
) -> float:
    """Find the depth that minimizes weighted alignment error for a tile.

    Tests multiple depth candidates and returns the best one.
    """
    N = warped_tiles.shape[0]
    th, tw = ref_tile.shape
    best_depth = base_depth
    best_cost = np.inf

    # Reference tile pixels
    ref_valid = ref_tile.copy()

    for z in depth_candidates:
        if abs(z - base_depth) < 0.1:
            # No correction needed at base depth
            cost = 0.0
            total_w = 0.0
            for i in range(N):
                if i == ref_idx:
                    continue
                w = weight_tiles[i] * mask_tiles[i]
                diff = np.abs(warped_tiles[i] - ref_valid)
                cost += (w * diff).sum()
                total_w += w.sum()
            if total_w > 0:
                cost /= total_w
        else:
            # Compute differential shift for this depth
            cost = 0.0
            total_w = 0.0
            for i in range(N):
                if i == ref_idx:
                    continue
                w = weight_tiles[i] * mask_tiles[i]
                if w.sum() < 1e-6:
                    continue
                # Compute homography correction: warp from base_depth to z
                H_base = compute_homography(
                    K_ref, K_all[i], R_rel[i], t_rel[i], base_depth
                )
                H_new = compute_homography(
                    K_ref, K_all[i], R_rel[i], t_rel[i], z
                )
                # Differential: H_new @ H_base^{-1}
                H_diff = H_new @ np.linalg.inv(H_base)
                # Apply only the translation part (small correction)
                # For a small tile, this is approximately a shift
                dx = H_diff[0, 2] / H_diff[2, 2]
                dy = H_diff[1, 2] / H_diff[2, 2]
                # Shift the warped tile
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(
                    warped_tiles[i], M, (tw, th),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                )
                diff = np.abs(shifted - ref_valid)
                cost += (w * diff).sum()
                total_w += w.sum()
            if total_w > 0:
                cost /= total_w

        if cost < best_cost:
            best_cost = cost
            best_depth = z

    return best_depth


def reconstruct_background_tiled(
    warped_images: np.ndarray, masks: np.ndarray,
    weights: np.ndarray,
    K: np.ndarray, R_rel: np.ndarray, t_rel: np.ndarray,
    focus_depth: float, ref_idx: int = 0,
    tile_size: int = 128, tile_overlap: int = 32,
    depth_range: float = 100.0, depth_steps: int = 11
) -> np.ndarray:
    """Tile-based background reconstruction with local depth refinement.

    Divides the image into overlapping tiles. For each tile:
    1. Search local depth to find best alignment
    2. Apply depth-corrected shifts
    3. Weighted composite favoring clean views

    Tiles are blended with Hann windows.

    Returns:
        (H, W) float32 reconstructed background.
    """
    N, H, W = warped_images.shape
    stride = tile_size - tile_overlap

    # Depth candidates
    if depth_steps > 1:
        depths = np.linspace(
            focus_depth - depth_range,
            focus_depth + depth_range,
            depth_steps
        )
        depths = np.maximum(depths, 10.0)
    else:
        depths = np.array([focus_depth])

    # Hann blending window
    hann_1d = np.hanning(tile_size).astype(np.float32)
    hann_2d = np.outer(hann_1d, hann_1d)

    output = np.zeros((H, W), dtype=np.float64)
    wbuf = np.zeros((H, W), dtype=np.float64)

    K_ref = K[ref_idx]
    n_tiles = 0

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            th, tw = y1 - y0, x1 - x0

            # Extract tiles
            ref_tile = warped_images[ref_idx, y0:y1, x0:x1]
            w_tiles = weights[:, y0:y1, x0:x1]
            m_tiles = masks[:, y0:y1, x0:x1]
            v_tiles = warped_images[:, y0:y1, x0:x1]

            # Local depth search
            best_z = _refine_tile_depth(
                ref_tile, v_tiles, w_tiles, m_tiles,
                depths, K_ref, K, R_rel, t_rel,
                ref_idx, (y0, x0), focus_depth
            )

            # Apply depth correction and composite
            tile_result = np.zeros((th, tw), dtype=np.float64)
            tile_wsum = np.zeros((th, tw), dtype=np.float64)

            for i in range(N):
                w = (w_tiles[i] * m_tiles[i]).astype(np.float64)
                w = w ** 2  # Squared weights sharpen selection
                if w.sum() < 1e-8:
                    continue

                if i == ref_idx or abs(best_z - focus_depth) < 0.1:
                    tile_val = v_tiles[i].astype(np.float64)
                else:
                    # Shift for depth correction
                    H_base = compute_homography(
                        K_ref, K[i], R_rel[i], t_rel[i], focus_depth
                    )
                    H_new = compute_homography(
                        K_ref, K[i], R_rel[i], t_rel[i], best_z
                    )
                    H_diff = H_new @ np.linalg.inv(H_base)
                    dx = H_diff[0, 2] / H_diff[2, 2]
                    dy = H_diff[1, 2] / H_diff[2, 2]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    tile_val = cv2.warpAffine(
                        v_tiles[i], M, (tw, th),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                    ).astype(np.float64)

                tile_result += w * tile_val
                tile_wsum += w

            # Normalize
            safe = tile_wsum > 1e-8
            tile_out = np.where(safe, tile_result / tile_wsum, ref_tile)

            # Hann blending
            win = hann_2d[:th, :tw].astype(np.float64)
            output[y0:y1, x0:x1] += tile_out * win
            wbuf[y0:y1, x0:x1] += win
            n_tiles += 1

    # Normalize accumulated tiles
    safe = wbuf > 1e-8
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(safe, output / wbuf, warped_images[ref_idx])
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def global_brightness_correction(
    result: np.ndarray,
    warped_images: np.ndarray, weights: np.ndarray,
    masks: np.ndarray,
    correction_sigma: float = 50.0,
    max_boost: float = 1.25
) -> np.ndarray:
    """Correct residual mesh attenuation via low-frequency flat-field.

    Compares the result to the theoretical clean ceiling and applies
    a smooth brightness correction.
    """
    masked = warped_images.copy()
    masked[masks < 0.5] = np.nan
    with np.errstate(all='ignore'):
        ceiling = np.nanpercentile(masked, 99, axis=0).astype(np.float32)
    ceiling = np.nan_to_num(ceiling, nan=0.0)

    ceil_blur = cv2.GaussianBlur(ceiling, (0, 0), correction_sigma)
    res_blur = cv2.GaussianBlur(result, (0, 0), correction_sigma)

    ratio = np.where(res_blur > 0.01, ceil_blur / res_blur, 1.0)
    ratio = np.clip(ratio, 1.0, max_boost).astype(np.float32)

    corrected = result * ratio
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def _v2_cache_path(output_dir: Path, focus_depth: float, ref_idx: int) -> Path:
    """Cache file path for warped images, masks, crop, and weights."""
    return output_dir / f"_v2_cache_d{focus_depth:.0f}_ref{ref_idx}.npz"


def run_defencing_v2(
    images: np.ndarray, K: np.ndarray, dist: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray,
    focus_depth: float = 750.0, ref_idx: int = 0,
    tile_size: int = 128, tile_overlap: int = 32,
    depth_range: float = 100.0, depth_steps: int = 11,
    sharpness_sigma: float = 3.0,
    consistency_sigma: float = 0.03,
    percentile_ref: float = 95.0,
    smooth_sigma: float = 5.0,
    brightness_correction: bool = True,
    max_boost: float = 1.25,
    warp_threads: int = 4,
    save_intermediates: bool = False,
    output_dir: Path = None
) -> np.ndarray:
    """Advanced defencing pipeline for dense defocused mesh removal.

    Designed for the case where a defocused metallic mesh covers the entire
    image. The mesh is specular (bright or dark) and out-of-focus, while the
    background is in-focus with fine texture.

    Pipeline:
      1. Undistort images
      2. Warp all views to reference at focus_depth
      3. Crop to overlap region
      4. Estimate per-view mesh weights (3-cue system)
      5. Tile-based reconstruction with local depth refinement
      6. Brightness correction for residual attenuation

    Steps 1-4 are cached to ``output_dir/_v2_cache_d{depth}_ref{idx}.npz``.
    When the cache exists and ``focus_depth`` / ``ref_idx`` match, those steps
    are skipped and the cached warped images, masks, crop, and weights are
    loaded directly.  This lets you iterate on step 5-6 parameters (tile size,
    depth search, brightness correction) without re-computing steps 1-4.

    Returns:
        (H, W) float32 deoccluded image in [0, 1].
    """
    N, H, W = images.shape
    output_size = (W, H)

    cache_file = None
    if output_dir is not None:
        cache_file = _v2_cache_path(output_dir, focus_depth, ref_idx)

    # Try loading cache
    if cache_file is not None and cache_file.exists():
        print(f"  [1-4] Loading cache: {cache_file.name}")
        cache = np.load(str(cache_file))
        warped_c = cache["warped"]
        masks_c = cache["masks"]
        w = cache["weights"]
        crop = cache["crop"]
        y0, y1, x0, x1 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
        print(f"    Crop: ({y0},{x0}) to ({y1},{x1}) = {x1-x0}x{y1-y0}")
    else:
        # Step 1: Undistort
        print("  [1/6] Undistorting...")
        images = undistort_images(images, K, dist)

        # Step 2: Warp all views
        print(f"  [2/6] Warping {N} views to depth {focus_depth:.0f} mm...")
        H_all = compute_all_homographies(K, R_rel, t_rel, focus_depth, ref_idx)
        warped = np.zeros((N, H, W), dtype=np.float32)
        masks = np.zeros((N, H, W), dtype=np.float32)
        for i in range(N):
            if i == ref_idx:
                warped[i] = images[i]
                masks[i] = 1.0
            else:
                warped[i] = warp_image(images[i], H_all[i], output_size)
                masks[i] = compute_validity_mask(
                    H_all[i], images[i].shape, output_size
                )

        # Step 3: Crop to overlap
        print("  [3/6] Cropping to overlap region...")
        y0, y1, x0, x1 = compute_overlap_crop(masks, margin=10)
        warped_c = warped[:, y0:y1, x0:x1].copy()
        masks_c = masks[:, y0:y1, x0:x1].copy()
        print(f"    Crop: ({y0},{x0}) to ({y1},{x1}) = {x1-x0}x{y1-y0}")

        # Step 4: Mesh weight estimation
        print("  [4/6] Estimating mesh weights...")
        w = estimate_mesh_weights(
            warped_c, masks_c,
            sharpness_sigma=sharpness_sigma,
            consistency_sigma=consistency_sigma,
            percentile_ref=percentile_ref,
            smooth_sigma=smooth_sigma
        )

        # Save cache
        if cache_file is not None:
            np.savez_compressed(
                str(cache_file),
                warped=warped_c, masks=masks_c, weights=w,
                crop=np.array([y0, y1, x0, x1])
            )
            print(f"    Saved cache: {cache_file.name}")

    if save_intermediates and output_dir is not None:
        w_vis = (w[ref_idx] / (w[ref_idx].max() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "mesh_weights_ref.png"), w_vis)
        w_mean = w.mean(axis=0)
        w_mean_vis = (w_mean / (w_mean.max() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "mesh_weights_mean.png"), w_mean_vis)

    # Step 5: Tiled reconstruction
    print("  [5/6] Tiled reconstruction...")
    result = reconstruct_background_tiled(
        warped_c, masks_c, w,
        K, R_rel, t_rel,
        focus_depth, ref_idx,
        tile_size=tile_size, tile_overlap=tile_overlap,
        depth_range=depth_range, depth_steps=depth_steps
    )

    # Step 6: Brightness correction
    if brightness_correction:
        print("  [6/6] Brightness correction...")
        result = global_brightness_correction(
            result, warped_c, w, masks_c, max_boost=max_boost
        )
    else:
        print("  [6/6] Brightness correction skipped.")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Plane-Sweep Stereo
# ═══════════════════════════════════════════════════════════════════════════════

def select_cameras_for_depth(
    t_rel: np.ndarray, z: float,
    sensor_width: int = 1440, f_px: float = 1739.13,
    max_disp_fraction: float = 0.9
) -> list:
    """Return camera indices whose disparity fits within the sensor.

    At very small z, far cameras produce disparity exceeding sensor width.
    """
    N = t_rel.shape[0]
    max_disp = max_disp_fraction * sensor_width
    valid = []
    for i in range(N):
        baseline = np.sqrt(t_rel[i, 0]**2 + t_rel[i, 1]**2)
        disparity = f_px * baseline / z
        if disparity < max_disp:
            valid.append(i)
    return valid


def compute_variance_cost(
    ref: np.ndarray, warped: list, masks: list,
    window_size: int = 5, min_valid: int = 3
) -> np.ndarray:
    """Variance-based photometric cost over a local window.

    Computes the variance of pixel intensities across views at each location,
    averaged over a local window. Lower variance = better depth match.

    Returns:
        (H, W) float32 cost map. Lower = better.
    """
    H, W = ref.shape
    N = len(warped)

    # Stack all images including reference
    all_imgs = np.zeros((N + 1, H, W), dtype=np.float32)
    all_masks = np.zeros((N + 1, H, W), dtype=np.float32)
    all_imgs[0] = ref
    all_masks[0] = 1.0
    for i in range(N):
        all_imgs[i + 1] = warped[i]
        all_masks[i + 1] = masks[i]

    # Count valid cameras per pixel
    valid_count = np.sum(all_masks, axis=0)

    # Compute mean and variance only where enough cameras are valid
    weighted_sum = np.sum(all_imgs * all_masks, axis=0)
    safe_count = np.maximum(valid_count, 1.0)
    mean_img = weighted_sum / safe_count

    # Variance across views
    diff_sq = (all_imgs - mean_img[np.newaxis]) ** 2 * all_masks
    variance = np.sum(diff_sq, axis=0) / safe_count

    # Box-filter aggregation
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    cost = cv2.filter2D(variance, -1, kernel / (window_size ** 2))

    # Mark pixels with too few valid cameras as worst cost
    cost[valid_count < min_valid] = 1.0

    return cost


def compute_ncc_cost(
    ref: np.ndarray, warped: list, masks: list,
    window_size: int = 5, min_valid: int = 3
) -> np.ndarray:
    """Normalized Cross-Correlation cost (1 - mean NCC).

    Returns:
        (H, W) float32 cost in [0, 2]. Lower = better.
    """
    H, W = ref.shape
    N = len(warped)
    eps = 1e-8
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    k_area = float(window_size ** 2)

    # Precompute reference local stats
    ref_mean = cv2.filter2D(ref, -1, kernel / k_area)
    ref_centered = ref - ref_mean
    ref_var = cv2.filter2D(ref_centered ** 2, -1, kernel / k_area)
    ref_std = np.sqrt(np.maximum(ref_var, eps))

    ncc_sum = np.zeros((H, W), dtype=np.float32)
    valid_count = np.zeros((H, W), dtype=np.float32)

    for i in range(N):
        w_img = warped[i]
        m = masks[i]
        w_masked = w_img * m

        w_mean = cv2.filter2D(w_masked, -1, kernel / k_area)
        w_centered = w_masked - w_mean
        w_var = cv2.filter2D(w_centered ** 2, -1, kernel / k_area)
        w_std = np.sqrt(np.maximum(w_var, eps))

        cross = cv2.filter2D(ref_centered * w_centered, -1, kernel / k_area)
        ncc = cross / (ref_std * w_std + eps)
        ncc_sum += ncc * m
        valid_count += m

    safe_count = np.maximum(valid_count, 1.0)
    mean_ncc = ncc_sum / safe_count
    cost = 1.0 - mean_ncc
    cost[valid_count < min_valid] = 1.0

    return cost


def compute_cost_volume_and_colors(
    images: np.ndarray, K: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray,
    depths: np.ndarray, ref_idx: int = 0,
    scale: float = 0.25, window_size: int = 5,
    cost_fn: str = "variance", min_valid_cameras: int = 3,
    sensor_width: int = 1440, f_px: float = 1739.13,
    warp_threads: int = 4
) -> tuple:
    """Compute cost volume AND per-plane colors at reduced resolution.

    For each depth plane, warps all source views and computes:
      - Photometric cost (variance or NCC)
      - Robust per-plane color via median of warped views
        (median rejects foreground outliers at background depths)

    Processes one plane at a time for memory efficiency.

    Args:
        images:    (N, H, W) float32 images in [0, 1].
        K:         (N, 3, 3) float64 intrinsics.
        R_rel:     (N, 3, 3) float64 relative rotations.
        t_rel:     (N, 3) float64 relative translations (mm).
        depths:    (D,) float32 depth values (mm).
        ref_idx:   Reference camera index.
        scale:     Downsample factor.
        window_size: Cost aggregation window.
        cost_fn:   "variance" or "ncc".
        min_valid_cameras: Minimum valid cameras per pixel.
        sensor_width: Sensor width in pixels.
        f_px:      Focal length in pixels.
        warp_threads: Parallel warp threads.

    Returns:
        cost_volume: (D, H_s, W_s) float32 cost volume.
        plane_colors: (D, H_s, W_s) float32 per-plane median colors.
    """
    N, H, W = images.shape
    D = len(depths)

    # Downsample images and adjust intrinsics
    images_s = downsample(images, scale)
    _, H_s, W_s = images_s.shape

    K_s = K.copy()
    K_s[:, 0, :] *= scale
    K_s[:, 1, :] *= scale

    cost_volume = np.zeros((D, H_s, W_s), dtype=np.float32)
    plane_colors = np.zeros((D, H_s, W_s), dtype=np.float32)
    output_size = (W_s, H_s)

    cost_func = compute_variance_cost if cost_fn == "variance" else compute_ncc_cost

    print(f"  Computing cost volume + plane colors: "
          f"{D} planes at {W_s}x{H_s} ({cost_fn})")
    for d in range(D):
        z = float(depths[d])

        # Select cameras that fit within sensor disparity
        valid_cams = select_cameras_for_depth(
            t_rel, z, sensor_width, f_px
        )
        # Exclude reference from warping (it's used as-is)
        source_cams = [c for c in valid_cams if c != ref_idx]

        if len(source_cams) < min_valid_cameras - 1:
            cost_volume[d] = 1.0
            plane_colors[d] = images_s[ref_idx]
            continue

        # Compute homographies for valid source cameras
        H_all = compute_all_homographies(K_s, R_rel, t_rel, z, ref_idx)

        # Warp source cameras
        warped = []
        masks = []
        for c in source_cams:
            w = warp_image(images_s[c], H_all[c], output_size)
            m = compute_validity_mask(H_all[c], images_s[c].shape, output_size)
            warped.append(w)
            masks.append(m)

        # Compute cost
        cost_volume[d] = cost_func(
            images_s[ref_idx], warped, masks,
            window_size, min_valid_cameras
        )

        # Compute per-plane color via median of all valid warped views.
        # At background depths, foreground pixels are misaligned across
        # views and rejected by median → clean background color.
        all_views = [images_s[ref_idx]]
        for i, c in enumerate(source_cams):
            # For invalid regions, substitute reference to avoid
            # black pixels pulling down the median
            filled = warped[i] * masks[i] + images_s[ref_idx] * (1.0 - masks[i])
            all_views.append(filled)
        stack = np.stack(all_views, axis=0)
        plane_colors[d] = np.median(stack, axis=0).astype(np.float32)

        if (d + 1) % 16 == 0 or d == D - 1:
            print(f"    plane {d+1}/{D}  z={z:.1f} mm  "
                  f"({len(source_cams)+1} cameras)")

    return cost_volume, plane_colors


# ═══════════════════════════════════════════════════════════════════════════════
# Alpha Estimation
# ═══════════════════════════════════════════════════════════════════════════════

def cost_to_alpha_softmax(
    cost_volume: np.ndarray, temperature: float = 0.1
) -> np.ndarray:
    """Convert cost volume to alpha via depth-axis softmax.

    Lower cost -> higher alpha. Each pixel sums to 1 across depth planes.
    Numerically stable (subtracts per-pixel max before exp).

    Returns:
        (D, H, W) float32 alpha volume summing to 1 along axis 0.
    """
    # Negate cost so lower cost = higher logit
    logits = -cost_volume / temperature

    # Numerical stability: subtract max per pixel
    logits_max = np.max(logits, axis=0, keepdims=True)
    logits_stable = logits - logits_max

    exp_logits = np.exp(logits_stable)
    sum_exp = np.sum(exp_logits, axis=0, keepdims=True)
    alphas = exp_logits / (sum_exp + 1e-10)

    return alphas.astype(np.float32)


def cost_to_alpha_wta_gaussian(
    cost_volume: np.ndarray, sigma: float = 1.5
) -> np.ndarray:
    """Winner-Takes-All with Gaussian spread across neighbouring planes.

    Returns:
        (D, H, W) float32 alpha volume summing to ~1 along axis 0.
    """
    D, H, W = cost_volume.shape
    best_plane = np.argmin(cost_volume, axis=0)  # (H, W)

    plane_indices = np.arange(D, dtype=np.float32).reshape(D, 1, 1)
    best_expanded = best_plane[np.newaxis].astype(np.float32)

    alphas = np.exp(-0.5 * ((plane_indices - best_expanded) / sigma) ** 2)
    alphas /= (np.sum(alphas, axis=0, keepdims=True) + 1e-10)

    return alphas.astype(np.float32)


def compute_confidence(cost_volume: np.ndarray) -> np.ndarray:
    """Per-pixel confidence from peak-to-valley ratio.

    confidence = (second_min - min) / (min + eps)
    Normalized to [0, 1] using 95th percentile as ceiling.

    Returns:
        (H, W) float32 confidence map in [0, 1].
    """
    D, H, W = cost_volume.shape
    sorted_cost = np.sort(cost_volume, axis=0)
    min_cost = sorted_cost[0]
    second_min = sorted_cost[1]

    eps = 1e-8
    conf = (second_min - min_cost) / (min_cost + eps)

    # Normalize using 95th percentile
    ceiling = np.percentile(conf, 95)
    if ceiling > 0:
        conf = np.clip(conf / ceiling, 0.0, 1.0)

    return conf.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# MPI Compositing
# ═══════════════════════════════════════════════════════════════════════════════

def composite_back_to_front(
    colors: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """Composite MPI planes back-to-front using the 'over' operator.

    Planes ordered: index 0 = farthest, index D-1 = nearest.
    C_out = C_d * alpha_d + C_out * (1 - alpha_d)

    Returns:
        (H, W) float32 composited image.
    """
    D = colors.shape[0]
    result = colors[0].copy()

    for d in range(1, D):
        a = alphas[d]
        result = colors[d] * a + result * (1.0 - a)

    return result.astype(np.float32)


def build_mpi_colors(ref_image: np.ndarray, n_planes: int) -> np.ndarray:
    """Broadcast reference image to all MPI depth planes.

    In CPU-only plane-sweep (no neural net), all planes share the same
    grayscale color; alpha determines which plane 'owns' each pixel.

    Returns:
        (D, H, W) float32 with ref_image replicated D times.
    """
    return np.broadcast_to(
        ref_image[np.newaxis], (n_planes,) + ref_image.shape
    ).copy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Foreground Removal
# ═══════════════════════════════════════════════════════════════════════════════

def remove_foreground_hard(
    alphas: np.ndarray, depths: np.ndarray, z_threshold: float
) -> np.ndarray:
    """Zero alpha for all planes with depth < z_threshold (hard cutoff).

    Returns:
        (D, H, W) float32 alpha volume with foreground planes zeroed.
    """
    result = alphas.copy()
    for d in range(len(depths)):
        if depths[d] < z_threshold:
            result[d] = 0.0
    return result


def remove_foreground_soft(
    alphas: np.ndarray, depths: np.ndarray,
    z_soft_near: float, z_soft_far: float
) -> np.ndarray:
    """Smooth ramp foreground removal.

    scale = 0   if depth < z_soft_near
    scale = 1   if depth > z_soft_far
    scale = smoothstep(linear ramp) between

    Returns:
        (D, H, W) float32 alpha volume with smooth foreground attenuation.
    """
    result = alphas.copy()
    for d in range(len(depths)):
        z = depths[d]
        if z <= z_soft_near:
            result[d] = 0.0
        elif z < z_soft_far:
            t = (z - z_soft_near) / (z_soft_far - z_soft_near)
            # Smoothstep for natural transition
            scale = 3.0 * t * t - 2.0 * t * t * t
            result[d] *= scale
        # else: keep unchanged (scale = 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Classical Robust Compositing (fast alternatives to MPI)
# ═══════════════════════════════════════════════════════════════════════════════

def synthetic_aperture_median(warped_images: np.ndarray) -> np.ndarray:
    """Pixel-wise median across N aligned views.

    Effective for up to ~45% foreground coverage (8 of 16 views).

    Args:
        warped_images: (N, H, W) float32, all aligned to target depth.

    Returns:
        (H, W) float32 synthetic aperture image.
    """
    return np.median(warped_images, axis=0).astype(np.float32)


def synthetic_aperture_trimmed_mean(
    warped_images: np.ndarray, trim: float = 0.2
) -> np.ndarray:
    """Trimmed mean across aligned views: discard extreme values.

    With N=16 and trim=0.2, averages the middle 10 values per pixel.

    Returns:
        (H, W) float32 synthetic aperture image.
    """
    result = stats.trim_mean(warped_images, trim, axis=0)
    return result.astype(np.float32)


def synthetic_aperture_entropy(
    warped_images: np.ndarray, n_bins: int = 16
) -> np.ndarray:
    """Per-pixel Shannon entropy across N aligned views.

    Lower entropy = views agree = correct depth alignment.
    Used as a depth cost measure (Vaish et al., CVPR 2006).

    Returns:
        (H, W) float32 entropy map.
    """
    N, H, W = warped_images.shape
    entropy_map = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            values = warped_images[:, y, x]
            hist, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
            probs = hist.astype(np.float32) / N
            probs = probs[probs > 0]
            entropy_map[y, x] = -np.sum(probs * np.log2(probs))

    return entropy_map


# ═══════════════════════════════════════════════════════════════════════════════
# SOTA Foreground / Fence Removal (De-fencing)
# ═══════════════════════════════════════════════════════════════════════════════

def synthetic_aperture_ransac(
    warped_images: np.ndarray, masks: np.ndarray = None,
    n_iter: int = 50, inlier_threshold: float = 0.04
) -> np.ndarray:
    """RANSAC-based pixel compositing across N aligned views.

    For each pixel, finds the largest consensus set (inliers) and takes
    their mean.  Much more robust than median when >50% of views are
    occluded (e.g. dense fences).

    Reference: Vaish et al., "Reconstructing Occluded Surfaces Using
    Synthetic Apertures", CVPR 2006 — robust depth cost rationale.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        n_iter:        RANSAC iterations per pixel column.
        inlier_threshold: Max deviation to count as inlier.

    Returns:
        (H, W) float32 deoccluded image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    result = np.zeros((H, W), dtype=np.float32)

    # Vectorised over width for speed: process one row at a time
    for y in range(H):
        row_vals = warped_images[:, y, :]   # (N, W)
        row_mask = masks[:, y, :]           # (N, W)

        best_inlier_count = np.zeros(W, dtype=np.int32)
        best_inlier_sum = np.zeros(W, dtype=np.float32)
        best_inlier_cnt_f = np.zeros(W, dtype=np.float32)

        for _ in range(n_iter):
            # Random hypothesis: pick one view per pixel column
            hyp_idx = np.random.randint(0, N)
            hyp = row_vals[hyp_idx]  # (W,)

            # Count inliers across all views
            diffs = np.abs(row_vals - hyp[np.newaxis, :])  # (N, W)
            inlier = ((diffs < inlier_threshold) & (row_mask > 0.5))
            inlier_count = inlier.sum(axis=0)  # (W,)

            # Update where we found more inliers
            better = inlier_count > best_inlier_count
            if better.any():
                inlier_sum = np.sum(row_vals * inlier, axis=0)
                inlier_cnt_f = inlier_count.astype(np.float32)

                best_inlier_count[better] = inlier_count[better]
                best_inlier_sum[better] = inlier_sum[better]
                best_inlier_cnt_f[better] = inlier_cnt_f[better]

        safe_cnt = np.maximum(best_inlier_cnt_f, 1.0)
        result[y] = best_inlier_sum / safe_cnt

    return result


def synthetic_aperture_irls(
    warped_images: np.ndarray, masks: np.ndarray = None,
    n_iter: int = 10, sigma: float = 0.03
) -> np.ndarray:
    """Iterative Reweighted Least Squares compositing.

    Starts from the pixel-wise median, then iteratively downweights
    outlier views using a Cauchy-like weight function.  Converges to a
    robust M-estimator of the background intensity.

    Reference: Xue et al., "A Computational Approach for Obstruction-Free
    Photography", SIGGRAPH 2015 — iterative layer separation concept.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        n_iter:        Number of reweighting iterations.
        sigma:         Scale parameter for Cauchy weight function.

    Returns:
        (H, W) float32 deoccluded image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Initialise with weighted median
    estimate = np.median(warped_images, axis=0)  # (H, W)

    for _ in range(n_iter):
        # Residuals
        residuals = warped_images - estimate[np.newaxis]  # (N, H, W)
        # Cauchy weights: w = 1 / (1 + (r/sigma)^2)
        weights = 1.0 / (1.0 + (residuals / sigma) ** 2)
        weights *= masks

        w_sum = np.sum(weights, axis=0)
        safe_sum = np.maximum(w_sum, 1e-8)
        estimate = np.sum(weights * warped_images, axis=0) / safe_sum

    return np.clip(estimate, 0.0, 1.0).astype(np.float32)


def detect_foreground_multiview(
    warped_images: np.ndarray, masks: np.ndarray = None,
    deviation_threshold: float = 0.06, min_ratio: float = 0.3,
    dilate_px: int = 3
) -> np.ndarray:
    """Detect foreground (fence) pixels via multi-view inconsistency.

    At background focus depth, foreground elements are misaligned across
    views.  For each view, pixels deviating significantly from the robust
    consensus (median) are flagged as foreground.  The per-view masks are
    aggregated: a pixel is marked foreground if >= min_ratio of views
    disagree.

    This implements the core idea from Vaish et al. 2006 and extends it
    with morphological post-processing for thin structures (fence wires).

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        deviation_threshold: Intensity deviation to flag foreground.
        min_ratio:     Fraction of views that must disagree.
        dilate_px:     Morphological dilation radius for thin structures.

    Returns:
        (H, W) float32 foreground probability map in [0, 1].
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Robust background estimate (median across views)
    bg_estimate = np.median(warped_images, axis=0)  # (H, W)

    # Per-view deviation from median
    deviations = np.abs(warped_images - bg_estimate[np.newaxis])  # (N, H, W)
    is_outlier = (deviations > deviation_threshold) & (masks > 0.5)

    # Fraction of views that flag this pixel as foreground
    valid_count = np.sum(masks > 0.5, axis=0).astype(np.float32)
    outlier_count = np.sum(is_outlier, axis=0).astype(np.float32)
    fg_ratio = outlier_count / np.maximum(valid_count, 1.0)

    # Binary foreground mask
    fg_mask = (fg_ratio >= min_ratio).astype(np.uint8)

    # Morphological operations: dilate to catch fence wire edges,
    # then close small gaps within fence structure
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    return fg_mask.astype(np.float32)


def detect_foreground_gradient(
    warped_images: np.ndarray, masks: np.ndarray = None,
    grad_threshold: float = 0.08, consistency_threshold: float = 0.5,
    dilate_px: int = 3
) -> np.ndarray:
    """Detect fence-like thin foreground via gradient inconsistency.

    Fences produce strong, thin edges that shift position across views
    at background focus depth.  This detector finds pixels where gradient
    magnitude is high but gradient direction/position is inconsistent
    across views — a hallmark of out-of-focus foreground structures.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        grad_threshold: Minimum gradient magnitude to consider.
        consistency_threshold: Max gradient variance (normalised) for
            "consistent" — above this is flagged as foreground.
        dilate_px:     Morphological dilation for thin structures.

    Returns:
        (H, W) float32 foreground mask in [0, 1].
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Compute gradient magnitude per view
    grad_mags = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        gx = cv2.Sobel(warped_images[i], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(warped_images[i], cv2.CV_32F, 0, 1, ksize=3)
        grad_mags[i] = np.sqrt(gx ** 2 + gy ** 2)

    # Mean gradient magnitude across views
    valid_count = np.sum(masks > 0.5, axis=0).astype(np.float32)
    mean_grad = np.sum(grad_mags * masks, axis=0) / np.maximum(valid_count, 1.0)

    # Gradient variance across views (high = inconsistent edges)
    grad_diff_sq = ((grad_mags - mean_grad[np.newaxis]) ** 2) * masks
    grad_var = np.sum(grad_diff_sq, axis=0) / np.maximum(valid_count, 1.0)

    # Pixels with strong but inconsistent gradients = foreground
    has_strong_grad = mean_grad > grad_threshold
    has_inconsistent_grad = grad_var > (consistency_threshold * mean_grad ** 2 + 1e-8)

    fg_mask = (has_strong_grad & has_inconsistent_grad).astype(np.uint8)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    return fg_mask.astype(np.float32)


def composite_visibility_weighted(
    warped_images: np.ndarray, masks: np.ndarray = None,
    sigma: float = 0.04
) -> np.ndarray:
    """Visibility-weighted compositing across aligned views.

    Each view's contribution is weighted by its photometric consistency
    with the median.  Occluded views (foreground) get low weight.

    weight_i = exp(-|view_i - median|^2 / (2 * sigma^2)) * mask_i

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        sigma:         Consistency bandwidth.

    Returns:
        (H, W) float32 composited image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    median_img = np.median(warped_images, axis=0)  # (H, W)

    # Photometric consistency weights
    diffs = warped_images - median_img[np.newaxis]
    weights = np.exp(-0.5 * (diffs / sigma) ** 2) * masks  # (N, H, W)

    w_sum = np.sum(weights, axis=0)
    safe_sum = np.maximum(w_sum, 1e-8)
    result = np.sum(weights * warped_images, axis=0) / safe_sum

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def inpaint_foreground_multiview(
    warped_images: np.ndarray, masks: np.ndarray,
    fg_mask: np.ndarray, sigma: float = 0.04
) -> np.ndarray:
    """Reconstruct foreground-masked pixels using unoccluded views.

    For pixels flagged as foreground, uses visibility-weighted
    compositing of only the "background" views (those that agree with
    the consensus).  For non-foreground pixels, uses a simple
    visibility-weighted composite.

    Falls back to OpenCV inpainting for any remaining holes where no
    view provides a clean background sample.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks.
        fg_mask:       (H, W) float32 foreground mask in [0, 1].
        sigma:         Consistency bandwidth for weighting.

    Returns:
        (H, W) float32 reconstructed image.
    """
    N, H, W = warped_images.shape
    median_img = np.median(warped_images, axis=0)

    # Per-view, per-pixel: is this view showing background?
    diffs = np.abs(warped_images - median_img[np.newaxis])
    # For foreground pixels, only trust views close to median
    bg_agreement = (diffs < sigma * 3.0).astype(np.float32) * masks

    # Weights: photometric consistency
    weights_all = np.exp(-0.5 * (diffs / sigma) ** 2) * masks
    weights_bg = weights_all * bg_agreement

    # For foreground pixels: use only bg-agreeing views
    # For background pixels: use all views with consistency weights
    fg_2d = fg_mask[np.newaxis]  # (1, H, W)
    weights = weights_bg * fg_2d + weights_all * (1.0 - fg_2d)

    w_sum = np.sum(weights, axis=0)
    safe_sum = np.maximum(w_sum, 1e-8)
    result = np.sum(weights * warped_images, axis=0) / safe_sum

    # Inpaint remaining holes (where no view has valid background)
    hole_mask = ((w_sum < 1e-4) & (fg_mask > 0.5)).astype(np.uint8)
    if hole_mask.sum() > 0:
        result_u8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        inpainted = cv2.inpaint(result_u8, hole_mask, 5, cv2.INPAINT_TELEA)
        result_inpainted = inpainted.astype(np.float32) / 255.0
        # Merge: use inpainted only in hole regions
        result = result * (1.0 - hole_mask.astype(np.float32)) + \
                 result_inpainted * hole_mask.astype(np.float32)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Fine Mesh / Net Removal
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_mesh_period(image: np.ndarray, min_period: int = 5,
                         max_period: int = 40) -> float:
    """Estimate the dominant periodic structure (mesh) period via FFT.

    Computes the 2D power spectrum, masks out the DC and very-low
    frequencies, then finds the dominant peak distance from centre.

    Args:
        image:      (H, W) float32 image.
        min_period: Minimum plausible mesh period in pixels.
        max_period: Maximum plausible mesh period in pixels.

    Returns:
        Estimated mesh period in pixels.
    """
    H, W = image.shape
    # Use central crop to avoid border effects
    ch, cw = H // 2, W // 2
    crop = image[H // 4:H // 4 + ch, W // 4:W // 4 + cw]

    fft = np.fft.fft2(crop)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    cy, cx = ch // 2, cw // 2

    # Mask out DC and very low frequencies
    Y, X = np.ogrid[:ch, :cw]
    dist_from_center = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    min_freq = ch / max_period  # frequency = size / period
    max_freq = ch / min_period

    mask = (dist_from_center >= min_freq) & (dist_from_center <= max_freq)
    power_masked = power * mask

    if power_masked.max() < 1e-10:
        return float(min_period + max_period) / 2.0

    # Find peak
    peak_idx = np.unravel_index(np.argmax(power_masked), power_masked.shape)
    peak_dist = np.sqrt((peak_idx[0] - cy) ** 2 + (peak_idx[1] - cx) ** 2)

    if peak_dist < 1:
        return float(min_period + max_period) / 2.0

    period = float(ch) / peak_dist
    return np.clip(period, min_period, max_period)


def detect_mesh_highpass(
    warped_images: np.ndarray, masks: np.ndarray = None,
    blur_sigma: float = 0.0, hf_threshold: float = 0.015,
    dilate_px: int = 1
) -> np.ndarray:
    """Detect fine mesh/net via high-frequency inconsistency across views.

    The mesh creates high-frequency texture that shifts position across
    views at background focus depth.  By extracting the high-frequency
    component (image - blurred) and computing its variance across views,
    we detect where mesh is present.

    If blur_sigma <= 0, it is auto-estimated from the mesh period.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        blur_sigma:    Gaussian blur sigma for low-pass. 0=auto.
        hf_threshold:  Mean |HF| threshold to flag as mesh.
        dilate_px:     Morphological dilation for thin structures.

    Returns:
        (H, W) float32 mesh mask in [0, 1].
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Auto-estimate blur sigma from mesh period
    if blur_sigma <= 0:
        period = estimate_mesh_period(warped_images[0])
        blur_sigma = period / 2.0
        print(f"    Auto mesh period: {period:.1f} px, blur sigma: {blur_sigma:.1f}")

    ksize = int(blur_sigma * 6) | 1  # ensure odd

    # Extract high-frequency per view
    hf = np.zeros_like(warped_images)
    for i in range(N):
        lf = cv2.GaussianBlur(warped_images[i], (ksize, ksize), blur_sigma)
        hf[i] = (warped_images[i] - lf) * masks[i]

    # Mean absolute HF across views — high where mesh texture exists
    hf_abs = np.abs(hf)
    valid_count = np.sum(masks > 0.5, axis=0).astype(np.float32)
    mean_hf = np.sum(hf_abs, axis=0) / np.maximum(valid_count, 1.0)

    # HF variance across views — high where mesh shifts between views
    hf_mean = np.sum(hf, axis=0) / np.maximum(valid_count, 1.0)
    hf_var = np.sum((hf - hf_mean[np.newaxis]) ** 2 * masks, axis=0) / \
             np.maximum(valid_count, 1.0)

    # Mesh = high HF content (texture present)
    mesh_mask = (mean_hf > hf_threshold).astype(np.uint8)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mesh_mask = cv2.dilate(mesh_mask, kernel, iterations=1)

    return mesh_mask.astype(np.float32)


def composite_lowpass_median(
    warped_images: np.ndarray, masks: np.ndarray = None,
    blur_sigma: float = 0.0
) -> np.ndarray:
    """Low-pass each view then take median — removes fine mesh.

    Each warped view is blurred to suppress the high-frequency mesh
    pattern.  The median of blurred views removes residual foreground.

    NOTE: No guided filter is applied here — guided filters using
    mesh-contaminated guides will reintroduce the mesh pattern.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        blur_sigma:    Gaussian blur sigma (0=auto from mesh period).

    Returns:
        (H, W) float32 mesh-free image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Auto-estimate blur sigma (use period/3 to avoid over-smoothing)
    if blur_sigma <= 0:
        period = estimate_mesh_period(warped_images[0])
        blur_sigma = max(2.0, period / 3.0)
        print(f"    Auto mesh period: {period:.1f} px, blur sigma: {blur_sigma:.1f}")

    ksize = int(blur_sigma * 6) | 1

    blurred = np.zeros_like(warped_images)
    for i in range(N):
        blurred[i] = cv2.GaussianBlur(warped_images[i], (ksize, ksize), blur_sigma)

    # Masked median: exclude invalid (black) warped regions
    # Replace invalid pixels with NaN, then use nanmedian
    blurred_masked = blurred.copy()
    blurred_masked[masks < 0.5] = np.nan
    with np.errstate(all='ignore'):
        result = np.nanmedian(blurred_masked, axis=0).astype(np.float32)
    # Fill any remaining NaN with the reference view (index 0)
    nan_mask = np.isnan(result)
    if nan_mask.any():
        result[nan_mask] = blurred[0][nan_mask]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def composite_spatial_median(
    warped_images: np.ndarray, masks: np.ndarray = None,
    kernel_size: int = 3
) -> np.ndarray:
    """Spatial median filter per view, then cross-view median.

    A spatial median filter at small kernel (3-5 px) removes thin
    wire-like structures (mesh lines) while preserving edges much
    better than Gaussian blur.  Cross-view median then removes any
    residual foreground.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        kernel_size:   Spatial median kernel (must be odd, default 3).

    Returns:
        (H, W) float32 mesh-free image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Spatial median filter per view — removes thin mesh wires
    filtered = np.zeros_like(warped_images)
    for i in range(N):
        img_u8 = np.clip(warped_images[i] * 255, 0, 255).astype(np.uint8)
        med_u8 = cv2.medianBlur(img_u8, kernel_size)
        filtered[i] = med_u8.astype(np.float32) / 255.0

    # Masked cross-view median
    filtered_masked = filtered.copy()
    filtered_masked[masks < 0.5] = np.nan
    with np.errstate(all='ignore'):
        result = np.nanmedian(filtered_masked, axis=0).astype(np.float32)
    nan_mask = np.isnan(result)
    if nan_mask.any():
        result[nan_mask] = filtered[0][nan_mask]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def composite_iterative_lowpass(
    warped_images: np.ndarray, masks: np.ndarray = None,
    blur_sigma: float = 3.0, n_passes: int = 2, median_kernel: int = 3
) -> np.ndarray:
    """Multi-pass mesh removal: spatial median + iterative LP median.

    Pipeline:
      1. Spatial median filter (removes thin mesh lines)
      2. LP + cross-view median at moderate sigma (N passes)

    This is more effective than single-pass LP at large sigma because
    each pass targets residual mesh at progressively lower amplitude.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        blur_sigma:    Per-pass Gaussian sigma (moderate: 3-5).
        n_passes:      Number of LP+median passes.
        median_kernel: Spatial median kernel size.

    Returns:
        (H, W) float32 mesh-free image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Pass 0: Spatial median per view
    current = np.zeros_like(warped_images)
    for i in range(N):
        img_u8 = np.clip(warped_images[i] * 255, 0, 255).astype(np.uint8)
        med_u8 = cv2.medianBlur(img_u8, median_kernel)
        current[i] = med_u8.astype(np.float32) / 255.0

    ksize = int(blur_sigma * 6) | 1

    for p in range(n_passes):
        # Gaussian blur each view
        blurred = np.zeros_like(current)
        for i in range(N):
            blurred[i] = cv2.GaussianBlur(
                current[i], (ksize, ksize), blur_sigma
            )

        # Masked cross-view median
        blurred_masked = blurred.copy()
        blurred_masked[masks < 0.5] = np.nan
        with np.errstate(all='ignore'):
            merged = np.nanmedian(blurred_masked, axis=0).astype(np.float32)
        nan_mask = np.isnan(merged)
        if nan_mask.any():
            merged[nan_mask] = blurred[0][nan_mask]

        # Replace all views with merged result for next pass
        for i in range(N):
            # Blend: keep valid original structure, merge in cleaned result
            current[i] = np.where(masks[i] > 0.5, merged, current[i])

    # Final cross-view median
    current_masked = current.copy()
    current_masked[masks < 0.5] = np.nan
    with np.errstate(all='ignore'):
        result = np.nanmedian(current_masked, axis=0).astype(np.float32)
    nan_mask = np.isnan(result)
    if nan_mask.any():
        result[nan_mask] = current[0][nan_mask]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def remove_mesh_notch_filter(
    image: np.ndarray, min_period: int = 5, max_period: int = 40,
    suppression: float = 0.05, n_harmonics: int = 4
) -> np.ndarray:
    """Remove periodic mesh pattern via FFT notch filtering.

    Detects dominant spectral peaks (mesh spatial frequency and
    harmonics) and suppresses them with Gaussian notch filters.

    Args:
        image:       (H, W) float32 image.
        min_period:  Minimum mesh period in pixels.
        max_period:  Maximum mesh period in pixels.
        suppression: Notch suppression strength (0=remove, 1=keep).
        n_harmonics: Number of harmonic peaks to suppress.

    Returns:
        (H, W) float32 image with periodic patterns removed.
    """
    H, W = image.shape

    # FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    # Frequency bounds for mesh
    min_freq = min(H, W) / max_period
    max_freq = min(H, W) / min_period

    # Find dominant peaks in the mesh frequency band
    band_mask = (dist_from_center >= min_freq) & (dist_from_center <= max_freq)
    mag_band = magnitude * band_mask

    # Find top peaks (exclude DC neighbourhood)
    notch_filter = np.ones((H, W), dtype=np.float64)

    # Get peak locations
    for harmonic in range(1, n_harmonics + 1):
        h_min_freq = min_freq * harmonic * 0.8
        h_max_freq = max_freq * harmonic * 1.2
        h_band = (dist_from_center >= h_min_freq) & (dist_from_center <= h_max_freq)
        h_mag = magnitude * h_band

        if h_mag.max() < 1e-6:
            continue

        # Find the strongest peak in this harmonic band
        # Use a smoothed version to find the peak region
        h_mag_smooth = cv2.GaussianBlur(h_mag.astype(np.float32), (5, 5), 1.0)
        peak_val = h_mag_smooth.max()
        threshold = peak_val * 0.3

        # Create notch: suppress all high-energy points in this band
        peak_region = (h_mag_smooth > threshold) & h_band
        # Gaussian suppression around peaks
        notch_width = max(2, int(min_freq * 0.3))
        peak_dilated = cv2.dilate(
            peak_region.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (notch_width * 2 + 1, notch_width * 2 + 1))
        ).astype(bool)

        notch_filter[peak_dilated] = suppression

    # Also suppress the symmetric counterpart (FFT symmetry)
    # Already handled since we find peaks in the full spectrum

    # Smooth the notch filter edges
    notch_filter = cv2.GaussianBlur(
        notch_filter.astype(np.float32), (3, 3), 0.5
    )

    # Apply notch filter
    fft_filtered = fft_shift * notch_filter
    fft_ishift = np.fft.ifftshift(fft_filtered)
    result = np.real(np.fft.ifft2(fft_ishift))

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def composite_mesh_removal(
    warped_images: np.ndarray, masks: np.ndarray = None,
    blur_sigma: float = 0.0, notch_suppression: float = 0.05,
    use_notch: bool = True, use_lowpass: bool = True
) -> np.ndarray:
    """Combined mesh removal: spatial median + LP median + notch filter.

    Pipeline:
      1. Spatial median per view (removes thin mesh wires)
      2. Low-pass each view → cross-view median → removes bulk mesh
      3. Apply notch filter to suppress any residual periodic patterns

    No guided filter is used — guides derived from mesh-contaminated
    images would reintroduce the mesh pattern.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        blur_sigma:    Gaussian blur sigma (0=auto).
        notch_suppression: FFT notch suppression strength.
        use_notch:     Apply FFT notch filter.
        use_lowpass:   Apply low-pass + median step.

    Returns:
        (H, W) float32 mesh-free image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    if use_lowpass:
        # Step 1: Spatial median per view (remove thin wires)
        prefiltered = np.zeros_like(warped_images)
        for i in range(N):
            img_u8 = np.clip(warped_images[i] * 255, 0, 255).astype(np.uint8)
            med_u8 = cv2.medianBlur(img_u8, 3)
            prefiltered[i] = med_u8.astype(np.float32) / 255.0

        # Step 2: LP + cross-view median
        result = composite_lowpass_median(prefiltered, masks, blur_sigma)
    else:
        # Just cross-view median without blur
        warp_masked = warped_images.copy()
        warp_masked[masks < 0.5] = np.nan
        with np.errstate(all='ignore'):
            result = np.nanmedian(warp_masked, axis=0).astype(np.float32)
        nan_mask = np.isnan(result)
        if nan_mask.any():
            result[nan_mask] = warped_images[0][nan_mask]

    # Step 3: Notch filter on residual periodic patterns
    if use_notch:
        period = estimate_mesh_period(warped_images[0])
        result = remove_mesh_notch_filter(
            result,
            min_period=max(3, int(period * 0.5)),
            max_period=int(period * 2.0),
            suppression=notch_suppression
        )

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Multiplicative Attenuation Model (Defocused Mesh Removal)
# ═══════════════════════════════════════════════════════════════════════════════

def composite_percentile(
    warped_images: np.ndarray, masks: np.ndarray = None,
    percentile: float = 90.0
) -> np.ndarray:
    """High-percentile compositing across aligned views.

    Since foreground mesh DARKENS pixels (multiplicative attenuation),
    taking a high percentile selects views where each pixel sees through
    a mesh gap — the brightest (least-attenuated) value.

    Unlike lowpass/median approaches, this preserves ALL background
    high-frequency detail because no blurring is applied.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        percentile:    Percentile to use (default 90). Higher = more
                       aggressive mesh removal but more noise sensitivity.

    Returns:
        (H, W) float32 deoccluded image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    warped_masked = warped_images.copy()
    warped_masked[masks < 0.5] = np.nan

    with np.errstate(all='ignore'):
        result = np.nanpercentile(warped_masked, percentile, axis=0).astype(np.float32)

    nan_mask = np.isnan(result)
    if nan_mask.any():
        result[nan_mask] = warped_images[0][nan_mask]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def composite_flatfield(
    warped_images: np.ndarray, masks: np.ndarray = None,
    percentile: float = 95.0, transmission_floor: float = 0.1,
    blur_sigma_bg: float = 3.0
) -> np.ndarray:
    """Flat-field correction for multiplicative mesh removal.

    Model: observed_i = background * transmission_i, where the defocused
    mesh creates a spatially-varying transmission T_i in (0, 1].

    Pipeline:
      1. Estimate clean background B via high percentile across views
      2. Smooth B to suppress noise in the estimate
      3. Per-view transmission: T_i = observed_i / B (clipped)
      4. Corrected: I_corrected_i = observed_i / T_i
      5. Weighted mean of corrected views

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        percentile:    Percentile for background estimate (default 95).
        transmission_floor: Min transmission to avoid blowup (default 0.1).
        blur_sigma_bg: Gaussian sigma to smooth background estimate (0=none).

    Returns:
        (H, W) float32 deoccluded image.
    """
    N, H, W = warped_images.shape
    if masks is None:
        masks = np.ones_like(warped_images)

    # Step 1: Estimate clean background via high percentile
    warped_masked = warped_images.copy()
    warped_masked[masks < 0.5] = np.nan
    with np.errstate(all='ignore'):
        bg_estimate = np.nanpercentile(
            warped_masked, percentile, axis=0
        ).astype(np.float32)

    nan_bg = np.isnan(bg_estimate)
    if nan_bg.any():
        bg_estimate[nan_bg] = warped_images[0][nan_bg]

    # Step 2: Smooth the background estimate
    if blur_sigma_bg > 0:
        ksize = int(blur_sigma_bg * 6) | 1
        bg_estimate = cv2.GaussianBlur(
            bg_estimate, (ksize, ksize), blur_sigma_bg
        )

    bg_safe = np.maximum(bg_estimate, 0.01)

    # Steps 3-5: Per-view correction and weighted average
    corrected_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for i in range(N):
        transmission = np.clip(
            warped_images[i] / bg_safe, transmission_floor, 1.0
        )
        corrected = warped_images[i] / transmission
        corrected_sum += corrected * masks[i]
        weight_sum += masks[i]

    result = corrected_sum / np.maximum(weight_sum, 1e-8)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def composite_hybrid(
    warped_images: np.ndarray, masks: np.ndarray = None,
    percentile: float = 90.0,
    guided_radius: int = 8, guided_eps: float = 0.01
) -> np.ndarray:
    """Hybrid: percentile guide + median base via guided filter.

    Combines strengths of two approaches:
      1. High-percentile: sharp, correct edges, but may have noise
      2. Median: smooth, robust brightness, but detail-lost
      3. Guided filter: transfers edges from percentile guide onto
         the smoother median base.

    Key: uses the mesh-FREE percentile result as guide, not the
    contaminated reference image.

    Args:
        warped_images: (N, H, W) float32 aligned views.
        masks:         (N, H, W) float32 validity masks (optional).
        percentile:    Percentile for clean guide (default 90).
        guided_radius: Guided filter radius (default 8).
        guided_eps:    Guided filter regularization (default 0.01).

    Returns:
        (H, W) float32 deoccluded image.
    """
    if masks is None:
        masks = np.ones_like(warped_images)

    # Sharp but potentially noisy guide
    clean_guide = composite_percentile(warped_images, masks, percentile)

    # Smooth, robust base
    warped_masked = warped_images.copy()
    warped_masked[masks < 0.5] = np.nan
    with np.errstate(all='ignore'):
        median_result = np.nanmedian(warped_masked, axis=0).astype(np.float32)
    nan_mask = np.isnan(median_result)
    if nan_mask.any():
        median_result[nan_mask] = warped_images[0][nan_mask]

    # Transfer edges from clean_guide onto median base
    try:
        result = cv2.ximgproc.guidedFilter(
            clean_guide, median_result, guided_radius, guided_eps
        )
    except AttributeError:
        print("    Warning: cv2.ximgproc unavailable, using percentile result")
        result = clean_guide

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Depth Map Extraction & Refinement
# ═══════════════════════════════════════════════════════════════════════════════

def extract_depth_map(cost_volume: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Extract depth map from cost volume via argmin.

    Returns:
        (H, W) float32 depth map in mm.
    """
    best_idx = np.argmin(cost_volume, axis=0)
    return depths[best_idx].astype(np.float32)


def fill_holes_inpaint(depth: np.ndarray, radius: float = 5.0) -> np.ndarray:
    """Fill zero-valued holes in depth map using Telea inpainting."""
    # Normalize to uint8 for inpainting
    d_min = depth[depth > 0].min() if np.any(depth > 0) else 0
    d_max = depth.max()
    if d_max - d_min < 1e-6:
        return depth

    d_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    mask = (depth == 0).astype(np.uint8)

    if mask.sum() == 0:
        return depth

    inpainted = cv2.inpaint(d_norm, mask, radius, cv2.INPAINT_TELEA)
    result = inpainted.astype(np.float32) / 255.0 * (d_max - d_min) + d_min
    # Keep original valid pixels
    result[depth > 0] = depth[depth > 0]
    return result


def refine_guided_filter(
    depth: np.ndarray, guide: np.ndarray,
    radius: int = 8, eps: float = 0.01
) -> np.ndarray:
    """Edge-preserving depth smoothing via guided filter.

    Uses the reference camera image as guide. Falls back to bilateral
    filter if cv2.ximgproc is unavailable.
    """
    try:
        refined = cv2.ximgproc.guidedFilter(guide, depth, radius, eps)
    except AttributeError:
        # Fallback to bilateral filter
        d_uint8 = np.clip(depth / depth.max() * 255, 0, 255).astype(np.uint8)
        d_filt = cv2.bilateralFilter(d_uint8, radius * 2, 75, 75)
        refined = d_filt.astype(np.float32) / 255.0 * depth.max()
    return refined.astype(np.float32)


def upsample_depth(
    depth_low: np.ndarray, guide_full: np.ndarray,
    scale: float = 4.0, radius: int = 8, eps: float = 0.01
) -> np.ndarray:
    """Upsample low-res depth map with guided joint upsampling."""
    H, W = guide_full.shape
    depth_up = cv2.resize(depth_low, (W, H), interpolation=cv2.INTER_LINEAR)
    return refine_guided_filter(depth_up, guide_full, radius, eps)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def colorize_depth(
    depth: np.ndarray, z_near: float, z_far: float, cmap: int = cv2.COLORMAP_TURBO
) -> np.ndarray:
    """Convert depth map to color-mapped BGR uint8 image."""
    d_norm = np.clip((depth - z_near) / (z_far - z_near), 0, 1)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(d_uint8, cmap)


def save_comparison(
    original: np.ndarray, deoccluded: np.ndarray, path: Path
):
    """Save side-by-side comparison of original and deoccluded images."""
    if original.ndim == 2:
        orig_bgr = cv2.cvtColor(
            np.clip(original * 255, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
    else:
        orig_bgr = np.clip(original * 255, 0, 255).astype(np.uint8)

    if deoccluded.ndim == 2:
        deoc_bgr = cv2.cvtColor(
            np.clip(deoccluded * 255, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
    else:
        deoc_bgr = np.clip(deoccluded * 255, 0, 255).astype(np.uint8)

    combined = np.hstack([orig_bgr, deoc_bgr])
    cv2.imwrite(str(path), combined)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_mpi_pipeline(
    images: np.ndarray, K: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray,
    depths: np.ndarray, ref_idx: int = 0,
    scale: float = 0.25, window_size: int = 5,
    cost_fn: str = "variance", temperature: float = 0.1,
    fg_removal: str = "soft",
    z_threshold: float = 120.0,
    z_soft_near: float = 80.0, z_soft_far: float = 150.0,
    sensor_width: int = 1440, f_px: float = 1739.13,
    warp_threads: int = 4,
    save_depth: bool = False, output_dir: Path = None,
    z_near: float = 50.0, z_far: float = 700.0
) -> np.ndarray:
    """Run the full MPI-based foreground removal pipeline.

    Returns:
        (H, W) float32 deoccluded image in [0, 1].
    """
    N, H, W = images.shape

    # Step 1: Cost volume + per-plane colors at reduced resolution
    print("[1/5] Computing cost volume and plane colors...")
    cost_volume, plane_colors = compute_cost_volume_and_colors(
        images, K, R_rel, t_rel, depths, ref_idx,
        scale, window_size, cost_fn, 3,
        sensor_width, f_px, warp_threads
    )

    # Step 2: Alpha estimation
    print("[2/5] Estimating alpha planes...")
    alphas = cost_to_alpha_softmax(cost_volume, temperature)

    # Step 3: Depth map extraction (optional save)
    if save_depth and output_dir is not None:
        print("  Extracting depth map...")
        depth_low = extract_depth_map(cost_volume, depths)
        depth_full = upsample_depth(depth_low, images[ref_idx],
                                     1.0 / scale)
        depth_color = colorize_depth(depth_full, z_near, z_far)
        cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
        print(f"  Saved depth map to {output_dir / 'depth_map.png'}")

    # Step 4: Foreground removal
    print("[3/5] Removing foreground...")
    if fg_removal == "hard":
        alphas_clean = remove_foreground_hard(alphas, depths, z_threshold)
    elif fg_removal == "soft":
        alphas_clean = remove_foreground_soft(alphas, depths,
                                              z_soft_near, z_soft_far)
    else:
        alphas_clean = alphas

    # Step 5: Composite at quarter-res, then upsample to full resolution
    print("[4/5] Compositing and upsampling...")

    # Composite at quarter resolution using per-plane median colors
    result_low = composite_back_to_front(plane_colors, alphas_clean)

    # Upsample to full resolution with guided filter
    result = cv2.resize(result_low, (W, H), interpolation=cv2.INTER_LINEAR)
    try:
        result = cv2.ximgproc.guidedFilter(
            images[ref_idx], result, radius=8, eps=0.01
        )
    except AttributeError:
        pass  # guided filter unavailable, keep bilinear upsample

    print("[5/5] Done.")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def run_defencing_pipeline(
    images: np.ndarray, K: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray,
    focus_depth: float, ref_idx: int = 0,
    composite_method: str = "ransac",
    detect_method: str = "multiview",
    deviation_threshold: float = 0.06,
    grad_threshold: float = 0.08,
    dilate_px: int = 3,
    ransac_iter: int = 50, ransac_threshold: float = 0.04,
    irls_iter: int = 10, irls_sigma: float = 0.03,
    blur_sigma: float = 0.0,
    percentile_q: float = 90.0,
    transmission_floor: float = 0.1,
    warp_threads: int = 4,
    save_masks: bool = False, output_dir: Path = None
) -> np.ndarray:
    """Advanced de-fencing pipeline: detect + remove foreground obstacles.

    Pipeline:
      1. Warp all views to background focus depth
      2. Detect foreground (fence) pixels via multi-view analysis
      3. Reconstruct clean background using robust compositing

    Compositing methods:
      - "ransac": RANSAC consensus (best for >50% occlusion)
      - "irls": Iterative Reweighted Least Squares (smooth M-estimator)
      - "visibility": Visibility-weighted compositing (fast, good baseline)
      - "inpaint": Detect fg mask + multi-view inpainting (sharpest)
      - "lowpass": Low-pass + median (best for fine mesh/net)
      - "mesh": Combined LP median + FFT notch (strongest mesh removal)

    Detection methods:
      - "multiview": Deviation from median across views
      - "gradient": Gradient inconsistency across views (thin structures)
      - "combined": Union of both detectors
      - "highpass": High-frequency inconsistency (best for fine mesh)

    Returns:
        (H, W) float32 deoccluded image in [0, 1].
    """
    N, H, W = images.shape
    output_size = (W, H)

    # Step 1: Warp all views to background focus depth
    print(f"  [1/3] Warping {N} views to focus depth {focus_depth:.0f} mm...")
    H_all = compute_all_homographies(K, R_rel, t_rel, focus_depth, ref_idx)

    warped_stack = np.zeros((N, H, W), dtype=np.float32)
    mask_stack = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        if i == ref_idx:
            warped_stack[i] = images[i]
            mask_stack[i] = 1.0
        else:
            warped_stack[i] = warp_image(images[i], H_all[i], output_size)
            mask_stack[i] = compute_validity_mask(
                H_all[i], images[i].shape, output_size
            )

    # Step 2: Detect foreground
    print(f"  [2/3] Detecting foreground ({detect_method})...")
    if detect_method == "multiview":
        fg_mask = detect_foreground_multiview(
            warped_stack, mask_stack,
            deviation_threshold=deviation_threshold,
            dilate_px=dilate_px
        )
    elif detect_method == "gradient":
        fg_mask = detect_foreground_gradient(
            warped_stack, mask_stack,
            grad_threshold=grad_threshold,
            dilate_px=dilate_px
        )
    elif detect_method == "combined":
        fg_mv = detect_foreground_multiview(
            warped_stack, mask_stack,
            deviation_threshold=deviation_threshold,
            dilate_px=dilate_px
        )
        fg_grad = detect_foreground_gradient(
            warped_stack, mask_stack,
            grad_threshold=grad_threshold,
            dilate_px=dilate_px
        )
        fg_mask = np.maximum(fg_mv, fg_grad)
    elif detect_method == "highpass":
        fg_mask = detect_mesh_highpass(
            warped_stack, mask_stack,
            dilate_px=dilate_px
        )
    else:
        fg_mask = np.zeros((H, W), dtype=np.float32)

    fg_pct = fg_mask.mean() * 100
    print(f"    Foreground coverage: {fg_pct:.1f}%")

    if save_masks and output_dir is not None:
        fg_vis = (fg_mask * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "fg_mask.png"), fg_vis)
        print(f"    Saved foreground mask to {output_dir / 'fg_mask.png'}")

    # Step 3: Robust compositing
    print(f"  [3/3] Compositing ({composite_method})...")
    if composite_method == "ransac":
        result = synthetic_aperture_ransac(
            warped_stack, mask_stack,
            n_iter=ransac_iter, inlier_threshold=ransac_threshold
        )
    elif composite_method == "irls":
        result = synthetic_aperture_irls(
            warped_stack, mask_stack,
            n_iter=irls_iter, sigma=irls_sigma
        )
    elif composite_method == "visibility":
        result = composite_visibility_weighted(
            warped_stack, mask_stack
        )
    elif composite_method == "inpaint":
        result = inpaint_foreground_multiview(
            warped_stack, mask_stack, fg_mask
        )
    elif composite_method == "lowpass":
        result = composite_lowpass_median(
            warped_stack, mask_stack, blur_sigma=blur_sigma
        )
    elif composite_method == "mesh":
        result = composite_mesh_removal(
            warped_stack, mask_stack, blur_sigma=blur_sigma
        )
    elif composite_method == "spatial_median":
        result = composite_spatial_median(
            warped_stack, mask_stack, kernel_size=3
        )
    elif composite_method == "iterative":
        result = composite_iterative_lowpass(
            warped_stack, mask_stack,
            blur_sigma=blur_sigma if blur_sigma > 0 else 3.0,
            n_passes=2, median_kernel=3
        )
    elif composite_method == "percentile":
        result = composite_percentile(
            warped_stack, mask_stack, percentile=percentile_q
        )
    elif composite_method == "flatfield":
        result = composite_flatfield(
            warped_stack, mask_stack, percentile=percentile_q,
            transmission_floor=transmission_floor,
            blur_sigma_bg=blur_sigma if blur_sigma > 0 else 3.0
        )
    elif composite_method == "hybrid":
        result = composite_hybrid(
            warped_stack, mask_stack, percentile=percentile_q
        )
    else:
        result = synthetic_aperture_ransac(warped_stack, mask_stack)

    # Guided-filter refinement — SKIP for mesh/attenuation methods because
    # the reference image contains mesh and would reintroduce it
    if composite_method not in ("lowpass", "mesh", "spatial_median", "iterative",
                                 "percentile", "flatfield", "hybrid"):
        try:
            result = cv2.ximgproc.guidedFilter(
                images[ref_idx], result, radius=8, eps=0.01
            )
        except AttributeError:
            pass

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def run_classical_pipeline(
    images: np.ndarray, K: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray,
    focus_depth: float, ref_idx: int = 0,
    method: str = "median", trim: float = 0.2,
    warp_threads: int = 4
) -> np.ndarray:
    """Run classical robust compositing (median or trimmed mean).

    Warps all images to align at focus_depth and composites.

    Returns:
        (H, W) float32 result image in [0, 1].
    """
    N, H, W = images.shape
    output_size = (W, H)

    print(f"  Warping {N} images to depth {focus_depth:.0f} mm...")
    H_all = compute_all_homographies(K, R_rel, t_rel, focus_depth, ref_idx)

    warped_stack = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        if i == ref_idx:
            warped_stack[i] = images[i]
        else:
            warped_stack[i] = warp_image(images[i], H_all[i], output_size)

    if method == "median":
        result = synthetic_aperture_median(warped_stack)
    elif method == "trimmed_mean":
        result = synthetic_aperture_trimmed_mean(warped_stack, trim)
    else:
        result = np.mean(warped_stack, axis=0).astype(np.float32)

    return np.clip(result, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI-based Synthetic Aperture Rendering for foreground removal"
    )
    parser.add_argument("image_dir",
                        help="Directory containing camera images (e00.png ... e15.png)")
    parser.add_argument("--calib", required=True,
                        help="Path to calibration .npz file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <image_dir>/mpi_sar_output)")

    # Method selection
    parser.add_argument("--method", default="defencing_v2",
                        choices=["mpi", "median", "trimmed_mean", "mean",
                                 "defencing", "defencing_v2"],
                        help="Processing method (default: defencing_v2)")

    # Depth range
    parser.add_argument("--z-near", type=float, default=100.0,
                        help="Nearest depth plane in mm (default: 100)")
    parser.add_argument("--z-far", type=float, default=1000.0,
                        help="Farthest depth plane in mm (default: 1000)")
    parser.add_argument("--n-planes", type=int, default=64,
                        help="Number of depth planes (default: 64)")

    # Focus depth for classical methods
    parser.add_argument("--focus-depth", type=float, default=750.0,
                        help="Focus depth in mm for median/trimmed_mean (default: 750)")

    # Cost volume parameters
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Cost volume downsample factor (default: 0.25)")
    parser.add_argument("--window-size", type=int, default=5,
                        help="Cost aggregation window size (default: 5)")
    parser.add_argument("--cost-fn", default="variance",
                        choices=["variance", "ncc"],
                        help="Cost function (default: variance)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature (default: 0.1)")

    # Foreground removal
    parser.add_argument("--no-fg-removal", action="store_true",
                        help="Skip foreground removal")
    parser.add_argument("--soft-removal", action="store_true",
                        help="Use soft ramp removal (default for mpi method)")
    parser.add_argument("--z-threshold", type=float, default=120.0,
                        help="Hard removal depth cutoff in mm (default: 120)")
    parser.add_argument("--z-soft-near", type=float, default=80.0,
                        help="Soft ramp start depth in mm (default: 80)")
    parser.add_argument("--z-soft-far", type=float, default=150.0,
                        help="Soft ramp end depth in mm (default: 150)")

    # De-fencing parameters
    parser.add_argument("--composite", default="ransac",
                        choices=["ransac", "irls", "visibility", "inpaint",
                                 "lowpass", "mesh", "spatial_median",
                                 "iterative",
                                 "percentile", "flatfield", "hybrid"],
                        help="De-fencing compositing method (default: ransac)")
    parser.add_argument("--detect", default="combined",
                        choices=["multiview", "gradient", "combined",
                                 "highpass", "none"],
                        help="Foreground detection method (default: combined)")
    parser.add_argument("--deviation-threshold", type=float, default=0.06,
                        help="Multi-view deviation threshold (default: 0.06)")
    parser.add_argument("--grad-threshold", type=float, default=0.08,
                        help="Gradient inconsistency threshold (default: 0.08)")
    parser.add_argument("--dilate-px", type=int, default=3,
                        help="Morphological dilation for thin structures (default: 3)")
    parser.add_argument("--ransac-iter", type=int, default=50,
                        help="RANSAC iterations (default: 50)")
    parser.add_argument("--ransac-threshold", type=float, default=0.04,
                        help="RANSAC inlier threshold (default: 0.04)")
    parser.add_argument("--irls-iter", type=int, default=10,
                        help="IRLS iterations (default: 10)")
    parser.add_argument("--irls-sigma", type=float, default=0.03,
                        help="IRLS Cauchy scale parameter (default: 0.03)")
    parser.add_argument("--blur-sigma", type=float, default=0.0,
                        help="Gaussian blur sigma for lowpass/mesh methods "
                             "(0=auto from mesh period, default: 0)")
    parser.add_argument("--percentile-q", type=float, default=90.0,
                        help="Percentile for percentile/flatfield/hybrid methods "
                             "(default: 90)")
    parser.add_argument("--transmission-floor", type=float, default=0.1,
                        help="Minimum transmission for flatfield correction "
                             "(default: 0.1)")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save foreground detection masks")

    # Reference camera
    parser.add_argument("--ref-idx", type=int, default=0,
                        help="Reference camera index (default: 0)")
    parser.add_argument("--trim", type=float, default=0.2,
                        help="Trim fraction for trimmed_mean (default: 0.2)")
    parser.add_argument("--warp-threads", type=int, default=4,
                        help="Parallel warp threads (default: 4)")

    # Output options
    parser.add_argument("--save-depth", action="store_true",
                        help="Save colorized depth map")
    parser.add_argument("--save-comparison", action="store_true",
                        help="Save side-by-side before/after comparison")
    parser.add_argument("--undistort", action="store_true",
                        help="Undistort images before processing")

    # Defencing v2 parameters
    parser.add_argument("--tile-size", type=int, default=128,
                        help="Tile size for v2 tiled reconstruction")
    parser.add_argument("--tile-overlap", type=int, default=32,
                        help="Tile overlap for v2 tiled reconstruction")
    parser.add_argument("--depth-search-range", type=float, default=100.0,
                        help="Depth search range +/- mm around focus-depth")
    parser.add_argument("--depth-search-steps", type=int, default=11,
                        help="Number of depth hypotheses for local search")
    parser.add_argument("--brightness-correction", action="store_true",
                        default=True,
                        help="Apply brightness correction (default: on)")
    parser.add_argument("--no-brightness-correction",
                        action="store_false", dest="brightness_correction",
                        help="Disable brightness correction")
    parser.add_argument("--max-boost", type=float, default=1.25,
                        help="Maximum brightness boost factor")

    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a directory")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else image_dir / "mpi_sar_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration
    print("[Step 1] Loading calibration...")
    calib = load_calibration(args.calib)
    K_all = calib["K"]
    dist_all = calib["dist"]
    R_mats_all = calib["R_mats"]
    tvecs_all = calib["tvecs"]

    # Compute relative poses for all 16 cameras
    R_rel_all, t_rel_all = compute_relative_poses(
        R_mats_all, tvecs_all, args.ref_idx
    )

    # Load images
    print("[Step 2] Loading images...")
    images, cam_indices = load_images(image_dir)
    n_cameras = images.shape[0]

    # Sub-select calibration to match available cameras
    idx = np.array(cam_indices)
    K = K_all[idx]
    dist = dist_all[idx]
    R_rel = R_rel_all[idx]
    t_rel = t_rel_all[idx]

    # Map ref_idx from global camera index to local array index
    if args.ref_idx in cam_indices:
        ref_local = cam_indices.index(args.ref_idx)
    else:
        ref_local = 0
        print(f"  Warning: ref camera e{args.ref_idx:02d} not in loaded images, "
              f"using e{cam_indices[0]:02d} instead")

    print(f"  {n_cameras} images, shape {images.shape[1:]}  "
          f"ref=e{cam_indices[ref_local]:02d} (local idx {ref_local})")

    # Optional undistortion (v2 undistorts internally)
    if args.undistort and args.method != "defencing_v2":
        print("  Undistorting...")
        images = undistort_images(images, K, dist)

    # Estimate focal length from reference intrinsics
    f_px = float((K[ref_local, 0, 0] + K[ref_local, 1, 1]) / 2.0)
    sensor_w = images.shape[2]

    # Run selected method
    if args.method == "mpi":
        print("[Step 3] Running MPI pipeline...")
        depths = compute_depth_planes(args.z_near, args.z_far, args.n_planes)

        fg_removal = "none" if args.no_fg_removal else (
            "soft" if args.soft_removal else "soft"
        )

        result = run_mpi_pipeline(
            images, K, R_rel, t_rel, depths, ref_local,
            scale=args.scale, window_size=args.window_size,
            cost_fn=args.cost_fn, temperature=args.temperature,
            fg_removal=fg_removal,
            z_threshold=args.z_threshold,
            z_soft_near=args.z_soft_near, z_soft_far=args.z_soft_far,
            sensor_width=sensor_w, f_px=f_px,
            warp_threads=args.warp_threads,
            save_depth=args.save_depth, output_dir=output_dir,
            z_near=args.z_near, z_far=args.z_far
        )
    elif args.method == "defencing_v2":
        print("[Step 3] Running de-fencing v2 pipeline...")
        result = run_defencing_v2(
            images, K, dist, R_rel, t_rel,
            focus_depth=args.focus_depth,
            ref_idx=ref_local,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            depth_range=args.depth_search_range,
            depth_steps=args.depth_search_steps,
            brightness_correction=args.brightness_correction,
            max_boost=args.max_boost,
            warp_threads=args.warp_threads,
            save_intermediates=args.save_masks,
            output_dir=output_dir
        )
    elif args.method == "defencing":
        print("[Step 3] Running de-fencing pipeline...")
        result = run_defencing_pipeline(
            images, K, R_rel, t_rel,
            args.focus_depth, ref_local,
            composite_method=args.composite,
            detect_method=args.detect,
            deviation_threshold=args.deviation_threshold,
            grad_threshold=args.grad_threshold,
            dilate_px=args.dilate_px,
            ransac_iter=args.ransac_iter,
            ransac_threshold=args.ransac_threshold,
            irls_iter=args.irls_iter,
            irls_sigma=args.irls_sigma,
            blur_sigma=args.blur_sigma,
            percentile_q=args.percentile_q,
            transmission_floor=args.transmission_floor,
            warp_threads=args.warp_threads,
            save_masks=args.save_masks,
            output_dir=output_dir
        )
    else:
        print(f"[Step 3] Running classical {args.method} pipeline...")
        result = run_classical_pipeline(
            images, K, R_rel, t_rel,
            args.focus_depth, ref_local,
            method=args.method, trim=args.trim,
            warp_threads=args.warp_threads
        )

    # Save result
    result_path = output_dir / f"result_{args.method}.png"
    save_image(result, result_path)
    print(f"  Saved result to {result_path}")

    # Save comparison
    if args.save_comparison:
        comp_path = output_dir / f"comparison_{args.method}.png"
        save_comparison(images[ref_local], result, comp_path)
        print(f"  Saved comparison to {comp_path}")

    print("All done.")


if __name__ == "__main__":
    main()
