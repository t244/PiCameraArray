"""
Estimate a depth map from the full 4x4 camera array using calibration_results.npz
and display/save a 3D point cloud. Uses OpenCV for rectification and disparity, and
Open3D (if available) or Plotly for visualization.

Example:
python analyze/depth_from_array.py --calib calibration_results.npz \
    --images-dir collected_data/20260213_153815

Dependencies: opencv-python, numpy, (optional) open3d, (optional) plotly
"""

from pathlib import Path
import argparse
import numpy as np
import cv2
import sys
import os

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False
try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def load_calibration(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return dict(data)


def find_camera_prefixes(calib_dict):
    prefixes = set()
    for k in calib_dict.keys():
        if k.endswith('_camera_matrix'):
            prefixes.add(k[:-len('_camera_matrix')])
    return sorted(prefixes)


def find_pair_pose(calib_dict, left, right):
    """Find relative pose (R, T) from left camera to right camera.

    Checks exact key format '{left}_to_{right}_rotation_matrix' first,
    then the reverse direction (and inverts). Falls back to computing
    from per-camera board poses if direct pair is not found.
    """
    # Try forward: left_to_right
    rot_key = f"{left}_to_{right}_rotation_matrix"
    tvec_key = f"{left}_to_{right}_tvec"
    if rot_key in calib_dict and tvec_key in calib_dict:
        R = np.array(calib_dict[rot_key]).astype(np.float64)
        T = np.array(calib_dict[tvec_key]).astype(np.float64).reshape(3)
        return R, T

    # Try reverse: right_to_left (invert)
    rot_key_inv = f"{right}_to_{left}_rotation_matrix"
    tvec_key_inv = f"{right}_to_{left}_tvec"
    if rot_key_inv in calib_dict and tvec_key_inv in calib_dict:
        R_inv = np.array(calib_dict[rot_key_inv]).astype(np.float64)
        T_inv = np.array(calib_dict[tvec_key_inv]).astype(np.float64).reshape(3, 1)
        R = R_inv.T
        T = (-R @ T_inv).reshape(3)
        return R, T

    # Fallback: compute from per-camera board poses
    R, T = compute_relative_pose_from_board_poses(calib_dict, left, right)
    return R, T


def compute_relative_pose_from_board_poses(calib_dict, left, right):
    """Compute relative pose from per-camera board poses stored in calibration.

    Looks for '{prefix}_board_rvec' and '{prefix}_board_tvec' keys.
    Returns (R, T) or (None, None) if not possible.
    """
    def get_board_pose(prefix):
        rkey = f"{prefix}_board_rvec"
        tkey = f"{prefix}_board_tvec"
        if rkey in calib_dict and tkey in calib_dict:
            r = np.array(calib_dict[rkey]).astype(np.float64).reshape(3)
            t = np.array(calib_dict[tkey]).astype(np.float64).reshape(3)
            R, _ = cv2.Rodrigues(r)
            return R, t
        return None, None

    R1, T1 = get_board_pose(left)
    R2, T2 = get_board_pose(right)
    if R1 is None or R2 is None:
        return None, None

    # Relative rotation and translation from left to right
    R_rel = R2 @ R1.T
    t_rel = T2 - R_rel @ T1
    return R_rel, t_rel


def get_camera_matrices(calib_dict, prefix):
    km_key = f"{prefix}_camera_matrix"
    d_key = f"{prefix}_dist_coeffs"
    if km_key not in calib_dict or d_key not in calib_dict:
        raise KeyError(f"Calibration for {prefix} not found in npz")
    K = np.array(calib_dict[km_key]).astype(np.float64)
    dist = np.array(calib_dict[d_key]).astype(np.float64).reshape(-1)
    return K, dist


def rectify_pair(imgL, imgR, K1, D1, K2, D2, R, T, flags=cv2.CALIB_ZERO_DISPARITY):
    h, w = imgL.shape[:2]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T, flags=flags, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    rectL = cv2.remap(imgL, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    return rectL, rectR, Q


def compute_disparity(rectL_gray, rectR_gray, min_disp=0, num_disp=128, block_size=5):
    # Ensure num_disp is divisible by 16 as required by SGBM
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16

    window_size = block_size
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = matcher.compute(rectL_gray, rectR_gray).astype(np.float32) / 16.0
    return disp


def estimate_pose_from_images(left_img_path, right_img_path, K1, K2, images_undistorted=False, detector='ORB'):
    # Load
    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)
    if img1 is None or img2 is None:
        return None, None

    # Optionally undistort if distortion coeffs are provided in K? We assume images_undistorted means already undistorted
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if detector.upper() == 'ORB':
        det = cv2.ORB_create(5000)
    else:
        det = cv2.AKAZE_create()
    kp1, des1 = det.detectAndCompute(gray1, None)
    kp2, des2 = det.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 8:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Use K1 as camera matrix for essential matrix computation (assume similar intrinsics)
    K = K1
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    t = t.reshape(3)
    return R, t



def find_image_for_prefix(images_root, prefix):
    p = Path(images_root)
    # search common patterns
    patterns = [f"**/{prefix}*.png", f"**/{prefix}*.jpg", f"{prefix}*.png", f"{prefix}*.jpg", f"**/{prefix}/*.png", f"**/{prefix}/*.jpg"]
    for pat in patterns:
        res = list(p.glob(pat))
        if len(res) > 0:
            return str(res[0])
    return None


def classify_pair_direction(T):
    """Classify a stereo pair as 'horizontal', 'vertical', or 'diagonal' based on translation.

    Returns the direction string and the baseline magnitude.
    A pair is horizontal if |Tx| > 2*|Ty|, vertical if |Ty| > 2*|Tx|,
    otherwise diagonal.
    """
    tx, ty = abs(T[0]), abs(T[1])
    magnitude = np.linalg.norm(T)
    if tx > 2 * ty:
        return 'horizontal', magnitude
    elif ty > 2 * tx:
        return 'vertical', magnitude
    else:
        return 'diagonal', magnitude


def compute_stereo_depth_for_pair(imgL, imgR, K1, D1, K2, D2, R, T, direction,
                                   num_disp=192, block_size=5, images_undistorted=False):
    """Compute 3D points for one stereo pair.

    Returns (pts, colors) where pts is Nx3 float64 and colors is Nx3 uint8 BGR.
    3D points come directly from cv2.reprojectImageTo3D (uses Q-matrix intrinsics,
    not the original K), and colors are sampled from the rectified left image so
    that pixel positions are consistent.
    For vertical pairs, images are rotated 90 CW so SGBM can search along rows,
    then 3D points are rotated back to the original camera frame.
    """
    h = imgL.shape[0]
    D1_use = D1 if not images_undistorted else np.zeros_like(D1)
    D2_use = D2 if not images_undistorted else np.zeros_like(D2)

    if direction == 'vertical':
        imgL_rot = cv2.rotate(imgL, cv2.ROTATE_90_CLOCKWISE)
        imgR_rot = cv2.rotate(imgR, cv2.ROTATE_90_CLOCKWISE)

        # For a 90-CW rotation of an (h, w) image:
        #   new fx = old fy,  new cx = old cy
        #   new fy = old fx,  new cy = (h - 1) - old cx   ← h, not w
        K1_rot = np.array([
            [K1[1, 1], 0, K1[1, 2]],
            [0, K1[0, 0], h - 1 - K1[0, 2]],
            [0, 0, 1]
        ], dtype=np.float64)
        K2_rot = np.array([
            [K2[1, 1], 0, K2[1, 2]],
            [0, K2[0, 0], h - 1 - K2[0, 2]],
            [0, 0, 1]
        ], dtype=np.float64)

        # 90-CW image rotation corresponds to P = [[0,1,0],[-1,0,0],[0,0,1]] on 3D coords
        P = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
        R_rot = P @ R @ P.T
        T_rot = (P @ T.reshape(3)).reshape(3)

        rectL, rectR, Q = rectify_pair(imgL_rot, imgR_rot, K1_rot, D1_use, K2_rot, D2_use, R_rot, T_rot)
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disp = compute_disparity(grayL, grayR, min_disp=0, num_disp=num_disp, block_size=block_size)

        pts3 = cv2.reprojectImageTo3D(disp, Q)
        valid = np.isfinite(pts3[..., 2]) & (disp > disp.min())

        # Rotate 3D points back to original camera frame (inverse of P)
        P_inv = P.T
        pts3_back = (P_inv @ pts3.reshape(-1, 3).T).T.reshape(pts3.shape)

        # Sample colors from the rotated rectified left image at valid positions
        pts_out = pts3_back[valid]
        colors_out = rectL[valid]
        return pts_out, colors_out

    else:
        rectL, rectR, Q = rectify_pair(imgL, imgR, K1, D1_use, K2, D2_use, R, T)
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disp = compute_disparity(grayL, grayR, min_disp=0, num_disp=num_disp, block_size=block_size)
        pts3 = cv2.reprojectImageTo3D(disp, Q)
        valid = np.isfinite(pts3[..., 2]) & (disp > disp.min())
        pts_out = pts3[valid]
        colors_out = rectL[valid]
        return pts_out, colors_out


def compute_multi_view_depth(calib, images_root, ref_prefix, prefixes, num_disp=192, block_size=5, images_undistorted=False):
    """Collect 3D points from all stereo pairs against the reference camera.

    Each pair produces (pts Nx3, colors Nx3) via reprojectImageTo3D + rectified
    left image — so both geometry and colors are in a consistent coordinate frame.
    Diagonal pairs are skipped (SGBM cannot handle them).
    Returns (pts_all, colors_all) concatenated across all valid pairs.
    """
    ref_img_path = find_image_for_prefix(images_root, ref_prefix)
    if ref_img_path is None:
        raise FileNotFoundError(f'Reference image for {ref_prefix} not found in {images_root}')
    ref_img = cv2.imread(ref_img_path)
    h, w = ref_img.shape[:2]

    K_ref, D_ref = get_camera_matrices(calib, ref_prefix)

    all_pts = []
    all_colors = []
    pair_info = []

    for pref in prefixes:
        if pref == ref_prefix:
            continue
        img_path = find_image_for_prefix(images_root, pref)
        if img_path is None:
            print(f'Warning: image for {pref} not found, skipping')
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f'Warning: failed to read {img_path}, skipping')
            continue

        K2, D2 = get_camera_matrices(calib, pref)
        R, T = find_pair_pose(calib, ref_prefix, pref)
        if R is None or T is None:
            print(f'Pose for {ref_prefix} to {pref} not found in calibration, attempting estimate from images')
            R_est, T_est = estimate_pose_from_images(ref_img_path, img_path, K_ref, K2, images_undistorted)
            if R_est is None:
                print(f'Warning: relative pose for {ref_prefix} to {pref} could not be estimated, skipping')
                continue
            R, T = R_est, T_est

        direction, baseline = classify_pair_direction(T)
        print(f'  {ref_prefix}->{pref}: {direction} pair, baseline={baseline*1000:.1f}mm, T=[{T[0]*1000:.1f},{T[1]*1000:.1f},{T[2]*1000:.1f}]mm')

        if direction == 'diagonal':
            print(f'    Skipping diagonal pair (SGBM requires aligned baseline)')
            continue

        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        pts_pair, colors_pair = compute_stereo_depth_for_pair(
            ref_img, img, K_ref, D_ref, K2, D2, R, T, direction,
            num_disp=num_disp, block_size=block_size, images_undistorted=images_undistorted
        )

        print(f'    Valid points: {len(pts_pair)}')
        all_pts.append(pts_pair)
        all_colors.append(colors_pair)
        pair_info.append((pref, direction, baseline))

    if len(all_pts) == 0:
        raise RuntimeError('No valid 3D points from any stereo pair')

    print(f'\nCollected points from {len(pair_info)} pairs '
          f'({sum(1 for _, d, _ in pair_info if d == "horizontal")} horizontal, '
          f'{sum(1 for _, d, _ in pair_info if d == "vertical")} vertical)')

    return np.concatenate(all_pts, axis=0), np.concatenate(all_colors, axis=0)


# ---------------------------------------------------------------------------
# Multi-view stereo helpers (plane-sweep + bundle adjustment)
# ---------------------------------------------------------------------------

def get_all_camera_poses(calib, prefixes):
    """Extract (K, R, t) for each prefix from calibration.

    R and t come from the board pose: X_cam = R @ X_board + t  (meters).
    Returns dict[prefix] -> (K 3x3, R 3x3, t shape-(3,)).
    Skips prefixes that are missing board-pose keys.
    """
    result = {}
    for pref in prefixes:
        try:
            K, _ = get_camera_matrices(calib, pref)
            r_key = f'{pref}_board_rotation_matrix'
            t_key = f'{pref}_board_tvec'
            if r_key not in calib or t_key not in calib:
                print(f'Warning: board pose for {pref} not in calibration, skipping')
                continue
            R = np.array(calib[r_key]).astype(np.float64).reshape(3, 3)
            t = np.array(calib[t_key]).astype(np.float64).reshape(3)
            result[pref] = (K, R, t)
        except Exception as e:
            print(f'Warning: could not load pose for {pref}: {e}')
    return result


def detect_features_all(images_gray, method='ORB', n_features=3000):
    """Detect and describe features in every grayscale image.

    Returns (kps_all, descs_all) — parallel lists, one entry per image.
    """
    if method.upper() == 'ORB':
        detector = cv2.ORB_create(n_features)
    else:
        detector = cv2.AKAZE_create()
    kps_all, descs_all = [], []
    for gray in images_gray:
        kp, des = detector.detectAndCompute(gray, None)
        kps_all.append(kp)
        descs_all.append(des)
    return kps_all, descs_all


class _UnionFind:
    """Simple path-compressed union-find for arbitrary hashable keys."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_tracks(kps_all, descs_all, max_dist=60, min_views=3):
    """Build multi-view feature tracks via union-find.

    Matches adjacent cameras in the 4×4 grid (horizontal and vertical
    neighbours only, to keep matching fast: 24 pairs instead of 120).
    Returns list of dicts  {cam_idx: (u, v)}  for tracks seen in ≥ min_views.
    """
    from collections import defaultdict
    N = len(kps_all)
    uf = _UnionFind()

    # For a 4×4 grid derive adjacent pairs; fall back to all pairs if N != 16
    if N == 16:
        pairs = []
        for i in range(16):
            row, col = divmod(i, 4)
            if col < 3:
                pairs.append((i, i + 1))       # horizontal neighbour
            if row < 3:
                pairs.append((i, i + 4))       # vertical neighbour
    else:
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i, j in pairs:
        if descs_all[i] is None or descs_all[j] is None:
            continue
        try:
            matches = bf.match(descs_all[i], descs_all[j])
        except Exception:
            continue
        for m in matches:
            if m.distance < max_dist:
                uf.union((i, m.queryIdx), (j, m.trainIdx))

    # Group observations into components
    components = defaultdict(list)
    for node in uf.parent:
        components[uf.find(node)].append(node)

    tracks = []
    for obs_list in components.values():
        cams = [c for c, _ in obs_list]
        if len(cams) != len(set(cams)):
            continue                             # ambiguous track — skip
        if len(cams) < min_views:
            continue
        track = {c: kps_all[c][ki].pt for c, ki in obs_list}
        tracks.append(track)
    return tracks


def triangulate_multiview(pts2d_per_cam, Ks, Rs, ts):
    """Linear DLT triangulation from N >= 2 views.

    pts2d_per_cam : sequence of (u, v) pixel coords, one per view
    Ks, Rs, ts    : intrinsics, rotation matrices and translations per view
                    (X_cam = R @ X_world + t)
    Returns X_world (3,) in world coordinates.
    """
    A = []
    for (u, v), K, R, t in zip(pts2d_per_cam, Ks, Rs, ts):
        P = K @ np.hstack([R, t.reshape(3, 1)])   # 3×4
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


# ---------------------------------------------------------------------------
# Bundle adjustment
# ---------------------------------------------------------------------------

def run_bundle_adjustment(Ks, Rs_init, ts_init, points3d_init, observations,
                          max_nfev=2000):
    """Full bundle adjustment via scipy least_squares (TRF + sparse Jacobian).

    Camera 0 is held fixed (gauge freedom).  All other cameras and all 3-D
    points are optimised to minimise the sum of squared reprojection errors,
    with a Huber loss for outlier robustness.

    Parameters
    ----------
    Ks          : list[ndarray 3×3]   — camera intrinsics (fixed)
    Rs_init     : list[ndarray 3×3]   — initial rotation matrices
    ts_init     : list[ndarray (3,)]  — initial translations (meters)
    points3d_init : ndarray (M, 3)   — initial 3-D points
    observations  : list of (cam_idx, pt_idx, u, v) tuples
    max_nfev      : max function evaluations

    Returns
    -------
    Rs_out, ts_out, points3d_out  (same shapes/types as inputs)
    """
    if not _HAS_SCIPY:
        print('scipy not available — skipping bundle adjustment')
        return Rs_init, ts_init, points3d_init

    N = len(Rs_init)
    M = len(points3d_init)
    n_obs = len(observations)

    # Pack camera params for cameras 1..N-1 (camera 0 is fixed)
    cam_params = []
    for i in range(1, N):
        rv, _ = cv2.Rodrigues(Rs_init[i])
        cam_params.extend(rv.flatten())
        cam_params.extend(ts_init[i].flatten())
    cam_params = np.array(cam_params, dtype=np.float64)

    pt_params = points3d_init.astype(np.float64).flatten()
    x0 = np.concatenate([cam_params, pt_params])

    n_cam_params = 6 * (N - 1)

    # Pre-extract observation arrays for vectorised residuals
    obs_ci = np.array([o[0] for o in observations], dtype=np.int32)
    obs_pi = np.array([o[1] for o in observations], dtype=np.int32)
    obs_u  = np.array([o[2] for o in observations], dtype=np.float64)
    obs_v  = np.array([o[3] for o in observations], dtype=np.float64)

    # Fixed intrinsics stacked for vectorised projection
    K_arr = np.stack(Ks, axis=0)     # (N, 3, 3)

    def unpack(x):
        # Returns R_arr (N,3,3), t_arr (N,3), pts3d (M,3)
        R_list = [Rs_init[0]]
        t_list = [ts_init[0]]
        for i in range(N - 1):
            rv = x[i * 6: i * 6 + 3]
            tv = x[i * 6 + 3: i * 6 + 6]
            R, _ = cv2.Rodrigues(rv)
            R_list.append(R)
            t_list.append(tv)
        R_arr  = np.stack(R_list, axis=0)   # (N, 3, 3)
        t_arr  = np.stack(t_list, axis=0)   # (N, 3)
        pts3d  = x[n_cam_params:].reshape(M, 3)
        return R_arr, t_arr, pts3d

    def residuals(x):
        R_arr, t_arr, pts3d = unpack(x)
        # Project every observation in one vectorised pass
        pts_obs = pts3d[obs_pi]                                   # (n_obs, 3)
        X_cam   = np.einsum('kij,kj->ki', R_arr[obs_ci], pts_obs) \
                  + t_arr[obs_ci]                                 # (n_obs, 3)
        proj    = np.einsum('kij,kj->ki', K_arr[obs_ci], X_cam)  # (n_obs, 3)
        depth   = proj[:, 2]
        valid   = depth > 1e-6
        u_proj  = np.where(valid, proj[:, 0] / np.where(valid, depth, 1.0), 1e3)
        v_proj  = np.where(valid, proj[:, 1] / np.where(valid, depth, 1.0), 1e3)
        res = np.empty(2 * n_obs, dtype=np.float64)
        res[0::2] = u_proj - obs_u
        res[1::2] = v_proj - obs_v
        return res

    # Sparse Jacobian structure
    sparsity = lil_matrix((2 * n_obs, len(x0)), dtype=np.int8)
    for k, (ci, pi, _, _) in enumerate(observations):
        r0, r1 = 2 * k, 2 * k + 1
        if ci > 0:
            c_start = 6 * (ci - 1)
            sparsity[r0, c_start: c_start + 6] = 1
            sparsity[r1, c_start: c_start + 6] = 1
        p_start = n_cam_params + 3 * pi
        sparsity[r0, p_start: p_start + 3] = 1
        sparsity[r1, p_start: p_start + 3] = 1

    print(f'  BA: {N} cameras, {M} points, {n_obs} observations')
    result = least_squares(
        residuals, x0,
        jac_sparsity=sparsity,
        method='trf',
        loss='huber',
        x_scale='jac',
        max_nfev=max_nfev,
        verbose=1,
    )
    R_arr_out, t_arr_out, pts_out = unpack(result.x)
    cost_before = np.sqrt(np.mean(residuals(x0) ** 2))
    cost_after  = np.sqrt(np.mean(result.fun ** 2))
    print(f'  BA RMS reprojection: {cost_before:.2f} px -> {cost_after:.2f} px')
    Rs_out = [R_arr_out[i] for i in range(N)]
    ts_out = [t_arr_out[i] for i in range(N)]
    return Rs_out, ts_out, pts_out


# ---------------------------------------------------------------------------
# Plane-sweep stereo
# ---------------------------------------------------------------------------

def compute_ncc(img_a_gray, img_b_gray, r):
    """Compute local normalised cross-correlation (NCC) between two float32 maps.

    Uses box-filter approximations for efficiency.
    Returns map in [-1, 1]; values near +1 indicate strong photometric match.
    """
    ksize = (2 * r + 1, 2 * r + 1)
    A = img_a_gray.astype(np.float32)
    B = img_b_gray.astype(np.float32)

    mu_a  = cv2.boxFilter(A,       -1, ksize)
    mu_b  = cv2.boxFilter(B,       -1, ksize)
    mu_a2 = cv2.boxFilter(A * A,   -1, ksize)
    mu_b2 = cv2.boxFilter(B * B,   -1, ksize)
    mu_ab = cv2.boxFilter(A * B,   -1, ksize)

    sigma_a2  = np.maximum(mu_a2 - mu_a * mu_a, 0.0)
    sigma_b2  = np.maximum(mu_b2 - mu_b * mu_b, 0.0)
    sigma_ab  = mu_ab - mu_a * mu_b

    ncc = sigma_ab / (np.sqrt(sigma_a2 * sigma_b2) + 1e-6)
    return np.clip(ncc, -1.0, 1.0)


def plane_sweep_stereo(images_color, Ks, Rs, ts, ref_idx,
                       depth_min_mm, depth_max_mm, num_depths, patch_radius):
    """Dense depth via plane-sweep photo-consistency across all cameras.

    For each depth hypothesis d, every source camera is warped into the
    reference view with a plane-induced homography, and NCC is computed
    against the reference.  The depth with the highest mean NCC wins.

    Parameters
    ----------
    images_color : list of BGR uint8 images (all same size, already undistorted)
    Ks, Rs, ts   : intrinsics and board poses per camera
                   (X_cam = R @ X_board + t, meters)
    ref_idx      : index of the reference camera in the lists
    depth_*      : depth range in mm

    Returns
    -------
    depth_map   : (H, W) float32  — winning depth in mm per pixel
    confidence  : (H, W) float32  — mean NCC score of the winner  ∈ [-1, 1]
    """
    ref_img  = images_color[ref_idx]
    H, W     = ref_img.shape[:2]
    K_ref    = Ks[ref_idx]
    R_ref    = Rs[ref_idx]
    t_ref    = ts[ref_idx]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Pre-convert sources to gray
    src_grays = []
    for i, img in enumerate(images_color):
        if i == ref_idx:
            src_grays.append(None)
        else:
            src_grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))

    depths_mm = np.linspace(depth_min_mm, depth_max_mm, num_depths)

    # cost_volume[y, x, d] = mean NCC across all source cameras at depth d
    cost_volume = np.full((H, W, num_depths), -1.0, dtype=np.float32)

    n = np.array([[0.0], [0.0], [1.0]])          # plane normal in ref frame
    n_sources = sum(1 for s in src_grays if s is not None)
    print(f'Plane-sweep: {num_depths} depth levels x {n_sources} source cameras')

    for d_idx, d_mm in enumerate(depths_mm):
        d_m = d_mm / 1000.0
        ncc_accum = np.zeros((H, W), dtype=np.float32)
        n_valid   = 0

        for i, src_gray in enumerate(src_grays):
            if src_gray is None:
                continue

            # Relative pose: source i → reference (X_ref = R_rel @ X_src + t_rel)
            R_rel = R_ref @ Rs[i].T
            t_rel = t_ref - R_rel @ ts[i]

            # Plane-induced homography (same sign convention as synthetic_aperture.py)
            # H = K_ref @ (R_rel - t_rel * n^T / d) @ K_src^{-1}
            try:
                H_mat = K_ref @ (R_rel - (t_rel.reshape(3, 1) @ n.T) / d_m) @ np.linalg.inv(Ks[i])
                H_mat = H_mat / H_mat[2, 2]
            except Exception:
                continue

            warped = cv2.warpPerspective(
                src_gray, H_mat, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )

            ncc = compute_ncc(ref_gray, warped, patch_radius)
            # Zero-warped regions give spurious NCC; mask them out
            ncc[warped == 0] = -1.0
            ncc_accum += ncc
            n_valid += 1

        if n_valid > 0:
            cost_volume[:, :, d_idx] = ncc_accum / n_valid

        if (d_idx + 1) % max(1, num_depths // 8) == 0:
            print(f'  depth {d_idx + 1}/{num_depths}: {d_mm:.0f} mm')

    best_idx   = np.argmax(cost_volume, axis=2)
    depth_map  = depths_mm[best_idx].astype(np.float32)
    confidence = cost_volume[
        np.arange(H)[:, None], np.arange(W)[None, :], best_idx
    ]
    return depth_map, confidence


def compute_mvs_plane_sweep(calib, images_root, ref_prefix, prefixes,
                             depth_min=200.0, depth_max=3000.0, num_depths=64,
                             scale=0.25, patch_radius=3, run_ba=True,
                             images_undistorted=False, confidence_threshold=0.3):
    """Top-level MVS: optional bundle adjustment + plane-sweep stereo.

    1. Loads and scales all images.
    2. Extracts camera poses from calibration (board poses).
    3. Optionally runs full bundle adjustment to refine extrinsics.
    4. Runs plane-sweep stereo across all cameras simultaneously.
    5. Back-projects the depth map to a 3-D point cloud.

    Returns (pts Nx3 float64 meters, colors Nx3 uint8 BGR) in reference
    camera frame.
    """
    # ---- 1. Load images -----------------------------------------------
    all_prefixes = [ref_prefix] + [p for p in prefixes if p != ref_prefix]

    ref_path = find_image_for_prefix(images_root, ref_prefix)
    if ref_path is None:
        raise FileNotFoundError(f'Reference image for {ref_prefix} not found in {images_root}')
    h_full, w_full = cv2.imread(ref_path).shape[:2]
    h_sc = max(1, int(h_full * scale))
    w_sc = max(1, int(w_full * scale))

    images_color   = []
    loaded_prefixes = []

    for pref in all_prefixes:
        img_path = find_image_for_prefix(images_root, pref)
        if img_path is None:
            print(f'Warning: image for {pref} not found, skipping')
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f'Warning: could not read {img_path}, skipping')
            continue
        if not images_undistorted:
            K_full, dist = get_camera_matrices(calib, pref)
            img = cv2.undistort(img, K_full, dist)
        img_sc = cv2.resize(img, (w_sc, h_sc))
        images_color.append(img_sc)
        loaded_prefixes.append(pref)

    n_cams = len(images_color)
    ref_idx = 0
    print(f'Loaded {n_cams} images at {w_sc}x{h_sc}')

    # ---- 2. Camera poses + scaled intrinsics --------------------------
    pose_dict = get_all_camera_poses(calib, loaded_prefixes)

    Ks, Rs, ts = [], [], []
    valid_indices = []
    for idx, pref in enumerate(loaded_prefixes):
        if pref not in pose_dict:
            print(f'Warning: no board pose for {pref}, skipping')
            continue
        K_full, R, t = pose_dict[pref]
        # Scale intrinsics to match resized image
        K_sc = K_full.copy()
        K_sc[0] *= scale      # fx, cx row
        K_sc[1] *= scale      # fy, cy row
        Ks.append(K_sc)
        Rs.append(R)
        ts.append(t)
        valid_indices.append(idx)

    images_color = [images_color[i] for i in valid_indices]
    n_cams = len(images_color)
    if n_cams < 2:
        raise RuntimeError('Need at least 2 cameras with board poses in calibration')

    # ---- 3. Optional bundle adjustment --------------------------------
    if run_ba:
        print('Running bundle adjustment...')
        images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_color]
        kps_all, descs_all = detect_features_all(images_gray)
        print(f'  Features detected: {[len(k) for k in kps_all]}')

        tracks = build_tracks(kps_all, descs_all)
        print(f'  Tracks (>=3 views): {len(tracks)}')

        if len(tracks) < 10:
            print('  Too few tracks for BA — using calibration poses directly')
        else:
            points3d_list = []
            observations  = []

            for track in tracks:
                cam_idxs = sorted(track.keys())
                pts2d    = [track[c] for c in cam_idxs]
                Ks_tr    = [Ks[c]   for c in cam_idxs]
                Rs_tr    = [Rs[c]   for c in cam_idxs]
                ts_tr    = [ts[c]   for c in cam_idxs]
                try:
                    X = triangulate_multiview(pts2d, Ks_tr, Rs_tr, ts_tr)
                except Exception:
                    continue
                # Keep only points in front of all cameras
                if any((Rs[c] @ X + ts[c])[2] <= 0 for c in cam_idxs):
                    continue
                pi = len(points3d_list)
                points3d_list.append(X)
                for c in cam_idxs:
                    u, v = track[c]
                    observations.append((c, pi, u, v))

            print(f'  Triangulated {len(points3d_list)} points, {len(observations)} obs')
            if len(points3d_list) >= 10:
                points3d_arr = np.array(points3d_list)
                Rs, ts, _ = run_bundle_adjustment(
                    Ks, Rs, ts, points3d_arr, observations
                )
            else:
                print('  Too few triangulated points — skipping BA')

    # ---- 4. Plane-sweep stereo ----------------------------------------
    print(f'Running plane-sweep: {depth_min}-{depth_max} mm, {num_depths} levels...')
    depth_map, confidence = plane_sweep_stereo(
        images_color, Ks, Rs, ts, ref_idx,
        depth_min, depth_max, num_depths, patch_radius
    )

    # ---- 5. Back-project to 3D ----------------------------------------
    valid = confidence > confidence_threshold
    ys, xs = np.where(valid)
    if len(xs) == 0:
        raise RuntimeError(
            f'No pixels passed confidence threshold {confidence_threshold}. '
            'Try lowering --confidence or widening --depth-min/--depth-max.'
        )

    depths_m = depth_map[valid].astype(np.float64) / 1000.0   # mm → m
    K_ref    = Ks[ref_idx]
    K_ref_inv = np.linalg.inv(K_ref)

    # X_cam = d * K^{-1} @ [u, v, 1]^T
    homo   = np.stack([xs.astype(np.float64),
                       ys.astype(np.float64),
                       np.ones(len(xs), np.float64)], axis=0)  # 3×N
    pts    = (K_ref_inv @ homo * depths_m[np.newaxis, :]).T    # N×3
    colors = images_color[ref_idx][ys, xs]                     # N×3 BGR

    print(f'Back-projected {len(pts)} 3D points '
          f'(confidence > {confidence_threshold})')
    return pts.astype(np.float64), colors.astype(np.uint8)


def save_ply(path, pts, colors):
    # colors expected in BGR 0-255
    verts = pts.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    # convert to RGB
    cols = cols[:, ::-1]
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(verts, cols):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def visualize_pointcloud(pts, colors, title='Point Cloud', downsample=50000, axis_range=None, point_size=1.0):
    n = len(pts)
    if n == 0:
        print('No points to show')
        return

    # Show bounds before cropping to help choose ranges
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    print(f'Point cloud bounds before crop: X[{mins[0]:.4f},{maxs[0]:.4f}] Y[{mins[1]:.4f},{maxs[1]:.4f}] Z[{mins[2]:.4f},{maxs[2]:.4f}] (n={n})')
    
    # Apply cropping with direct absolute values (no scaling)
    mask = np.ones(len(pts), dtype=bool)
    if axis_range is not None:
        xmin, xmax, ymin, ymax, zmin_a, zmax_a = axis_range
        mask &= (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
        mask &= (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        mask &= (pts[:, 2] >= zmin_a) & (pts[:, 2] <= zmax_a)

    if not np.any(mask):
        print('No points left after applying range filters')
        return

    pts = pts[mask]
    colors = colors[mask]

    mins2 = pts.min(axis=0)
    maxs2 = pts.max(axis=0)
    print(f'Point cloud bounds after crop: X[{mins2[0]:.4f},{maxs2[0]:.4f}] Y[{mins2[1]:.4f},{maxs2[1]:.4f}] Z[{mins2[2]:.4f},{maxs2[2]:.4f}] (n={len(pts)})')

    n = len(pts)
    if n > downsample:
        idx = np.random.choice(n, downsample, replace=False)
        pts = pts[idx]
        colors = colors[idx]

    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # colors in 0..1
        pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1] / 255.0)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = point_size
        vis.run()
        vis.destroy_window()
    else:
        if _HAS_PLOTLY:
            # Use Plotly scatter3d and save to standalone HTML (do not auto-open)
            rgb = (colors[:, ::-1]).astype(int)
            color_strs = [f'rgb({r},{g},{b})' for r, g, b in rgb]
            trace = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=point_size, color=color_strs, opacity=0.8)
            )
            fig = go.Figure(data=[trace])
            fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=title)
            out_html = Path('pointcloud.html').resolve()
            fig.write_html(str(out_html), auto_open=False)
            print(f'Point cloud saved to HTML: {out_html} - open this file in your browser.')
        else:
            print('Plotly is not installed. Install with `pip install plotly` to visualize the point cloud in the browser.')
            return


def parse_args():
    p = argparse.ArgumentParser(description='Estimate depth from all cameras using calibration results and show 3D view')
    p.add_argument('--calib', default='calibration_results.npz', help='Calibration npz file')
    p.add_argument('--images-dir', default='.', help='Root directory to search for per-camera images')
    p.add_argument('--ref-prefix', default=None, help='Reference camera prefix (defaults to first prefix in calib)')
    p.add_argument('--images-undistorted', action='store_true', help='Set if images are already undistorted (skips undistortion)')
    p.add_argument('--num-disp', type=int, default=192, help='Number of disparities for SGBM')
    p.add_argument('--block', type=int, default=5, help='Block size for SGBM')
    p.add_argument('--ply-out', default=None, help='Optional output PLY filename')
    p.add_argument('--downsample', type=int, default=50000, help='Max points to visualize')
    p.add_argument('--axis-range', nargs=6, type=float, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                   help='Crop the point cloud to xmin xmax ymin ymax zmin zmax before display')
    p.add_argument('--point-size', type=float, default=1.0, help='Point size for 3D visualization')
    # Plane-sweep MVS options
    p.add_argument('--method', choices=['stereo_pairs', 'plane_sweep'], default='stereo_pairs',
                   help='stereo_pairs: independent SGBM per pair (default); '
                        'plane_sweep: joint MVS via plane-sweep + optional BA')
    p.add_argument('--depth-min', type=float, default=200.0,
                   help='Minimum depth in mm for plane-sweep (default: 200)')
    p.add_argument('--depth-max', type=float, default=3000.0,
                   help='Maximum depth in mm for plane-sweep (default: 3000)')
    p.add_argument('--num-depths', type=int, default=64,
                   help='Number of depth hypotheses for plane-sweep (default: 64)')
    p.add_argument('--scale', type=float, default=0.25,
                   help='Image downscale factor for plane-sweep (default: 0.25)')
    p.add_argument('--patch-radius', type=int, default=3,
                   help='NCC patch half-size in pixels after scaling (default: 3)')
    p.add_argument('--no-ba', action='store_true',
                   help='Skip bundle adjustment in plane-sweep mode (use raw calibration poses)')
    p.add_argument('--confidence', type=float, default=0.3,
                   help='Min NCC confidence to keep a pixel in plane-sweep (default: 0.3)')
    return p.parse_args()

 
def main():
    args = parse_args()
    calib_path = Path(args.calib)
    if not calib_path.exists():
        print('Calibration file not found:', calib_path)
        sys.exit(1)

    calib = load_calibration(str(calib_path))
    prefixes = find_camera_prefixes(calib)
    print('Found camera prefixes in calibration:', prefixes)

    if len(prefixes) == 0:
        print('No camera prefixes found in calibration')
        sys.exit(1)

    ref_pref = args.ref_prefix if args.ref_prefix is not None else prefixes[0]
    images_undist = args.images_undistorted or ('_undist' in args.images_dir)
    print(f'Using all cameras with reference: {ref_pref} (images dir: {args.images_dir}) undistorted={images_undist}')

    if args.method == 'plane_sweep':
        pts, colors = compute_mvs_plane_sweep(
            calib, args.images_dir, ref_pref, prefixes,
            depth_min=args.depth_min, depth_max=args.depth_max,
            num_depths=args.num_depths, scale=args.scale,
            patch_radius=args.patch_radius, run_ba=not args.no_ba,
            images_undistorted=images_undist,
            confidence_threshold=args.confidence,
        )
    else:
        pts, colors = compute_multi_view_depth(
            calib, args.images_dir, ref_pref, prefixes,
            num_disp=args.num_disp, block_size=args.block, images_undistorted=images_undist
        )
    print(f'Generated {len(pts)} 3D points')

    if args.ply_out:
        save_ply(args.ply_out, pts, colors)
        print('Saved PLY to', args.ply_out)

    axis_range_vals = list(args.axis_range) if args.axis_range else None
    print(f'Visualizing with axis_range={axis_range_vals}')
    visualize_pointcloud(pts, colors, downsample=args.downsample, axis_range=axis_range_vals, point_size=args.point_size)


if __name__ == '__main__':
    main()
