"""
Estimate a depth map from a pair of cameras in the array using calibration_results.npz
and display/save a 3D point cloud. Uses OpenCV for rectification and disparity, and
Open3D (if available) or matplotlib for visualization.

Example:
python analyze/depth_from_array.py --calib calibration_results.npz \
    --left e00 --right e01 --left-img collected_data/20260213_153815/e00_20260213_153815/image.png \
    --right-img collected_data/20260213_153815/e01_20260213_154023/image.png \

Dependencies: opencv-python, numpy, matplotlib, (optional) open3d
"""

from pathlib import Path
import argparse
import numpy as np
import cv2
import sys
import os
import glob

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
    # Try to find rotation matrix and tvec for the pair.
    # Look for keys that contain both left and right.
    rot_key = None
    tvec_key = None
    for k in calib_dict.keys():
        if 'rotation_matrix' in k and left in k and right in k:
            rot_key = k
        if k.endswith('_tvec') and left in k and right in k:
            tvec_key = k

    # allow reversed order
    if rot_key is None or tvec_key is None:
        for k in calib_dict.keys():
            if 'rotation_matrix' in k and right in k and left in k:
                rot_key = k
            if k.endswith('_tvec') and right in k and left in k:
                tvec_key = k

    R = None
    T = None
    if rot_key is not None:
        R = np.array(calib_dict[rot_key]).astype(np.float64)
    if tvec_key is not None:
        T = np.array(calib_dict[tvec_key]).astype(np.float64).reshape(3)

    # If explicit pair pose is not available, try to compute from per-camera poses
    if R is None or T is None:
        R2, T2 = compute_relative_pose_from_single_poses(calib_dict, left, right)
        if R2 is not None and T2 is not None:
            return R2, T2

    return R, T


def compute_relative_pose_from_single_poses(calib_dict, left, right):
    """
    Compute relative pose (R, T) from single-camera poses stored in the calibration dict.
    Looks for keys like '{prefix}_rvec' and '{prefix}_tvec' or arrays of rvecs/tvecs and averages them.
    Returns (R, T) or (None, None) if not possible.
    """
    def get_mean_pose(prefix):
        # exact keys
        rkey = f"{prefix}_rvec"
        tkey = f"{prefix}_tvec"
        if rkey in calib_dict and tkey in calib_dict:
            r = np.array(calib_dict[rkey]).astype(np.float64).reshape(-1)
            t = np.array(calib_dict[tkey]).astype(np.float64).reshape(-1)
            if r.size == 3 and t.size == 3:
                R, _ = cv2.Rodrigues(r)
                return R, t

        # try plural keys (rvecs/tvecs) and average
        rkeyp = f"{prefix}_rvecs"
        tkeyp = f"{prefix}_tvecs"
        if rkeyp in calib_dict and tkeyp in calib_dict:
            r_arr = np.array(calib_dict[rkeyp]).astype(np.float64)
            t_arr = np.array(calib_dict[tkeyp]).astype(np.float64)
            if r_arr.size > 0 and t_arr.size > 0:
                mean_r = np.mean(r_arr, axis=0).reshape(3)
                mean_t = np.mean(t_arr, axis=0).reshape(3)
                R, _ = cv2.Rodrigues(mean_r)
                return R, mean_t

        # not found
        return None, None

    R1, T1 = get_mean_pose(left)
    R2, T2 = get_mean_pose(right)
    if R1 is None or R2 is None or T1 is None or T2 is None:
        return None, None

    # relative rotation and translation from left to right
    R_rel = R2 @ R1.T
    rvec_rel, _ = cv2.Rodrigues(R_rel)
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


def disparity_to_pointcloud(disp, Q, left_color):
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    mask = (disp > disp.min()) & np.isfinite(points_3d[..., 0])
    pts = points_3d[mask]
    colors = left_color[mask]
    return pts, colors


def find_image_for_prefix(images_root, prefix):
    p = Path(images_root)
    # search common patterns
    patterns = [f"**/{prefix}*.png", f"**/{prefix}*.jpg", f"{prefix}*.png", f"{prefix}*.jpg", f"**/{prefix}/*.png", f"**/{prefix}/*.jpg"]
    for pat in patterns:
        res = list(p.glob(pat))
        if len(res) > 0:
            return str(res[0])
    return None


def compute_multi_view_depth(calib, images_root, ref_prefix, prefixes, num_disp=192, block_size=5, images_undistorted=False):
    # load reference image for size and color
    ref_img_path = find_image_for_prefix(images_root, ref_prefix)
    if ref_img_path is None:
        raise FileNotFoundError(f'Reference image for {ref_prefix} not found in {images_root}')
    ref_img = cv2.imread(ref_img_path)
    h, w = ref_img.shape[:2]

    K_ref, D_ref = get_camera_matrices(calib, ref_prefix)

    depth_maps = []
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
        if images_undistorted:
            D2 = np.zeros_like(D2)
        R, T = find_pair_pose(calib, ref_prefix, pref)
        if R is None or T is None:
            # try to estimate pose from image matches if calibration lacks poses
            print(f'Pose for {ref_prefix} to {pref} not found in calibration, attempting estimate from images')
            R_est, T_est = estimate_pose_from_images(ref_img_path, img_path, K_ref, K2, images_undistorted)
            if R_est is None:
                print(f'Warning: relative pose for {ref_prefix} to {pref} could not be estimated, skipping')
                continue
            R, T = R_est, T_est

        # resize to match ref if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        rectL, rectR, Q = rectify_pair(ref_img, img, K_ref, D_ref if not images_undistorted else np.zeros_like(D_ref), K2, D2, R, T)
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disp = compute_disparity(grayL, grayR, min_disp=0, num_disp=num_disp, block_size=block_size)
        pts3 = cv2.reprojectImageTo3D(disp, Q)
        zmap = pts3[..., 2]
        # mask invalid
        mask = np.isfinite(zmap) & (disp > disp.min())
        zmap[~mask] = np.nan
        depth_maps.append(zmap)

    if len(depth_maps) == 0:
        raise RuntimeError('No valid depth maps computed from views')

    stack = np.stack(depth_maps, axis=0)
    depth_median = np.nanmedian(stack, axis=0)
    return depth_median, ref_img, K_ref


def depth_to_pointcloud_from_depthmap(depth_map, color_img, K):
    h, w = depth_map.shape[:2]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    ys, xs = np.where(np.isfinite(depth_map))
    zs = depth_map[ys, xs]
    xs_cam = (xs - cx) * zs / fx
    ys_cam = (ys - cy) * zs / fy
    pts = np.vstack((xs_cam, ys_cam, zs)).T
    colors = color_img[ys, xs]
    return pts, colors


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


def visualize_pointcloud(pts, colors, title='Point Cloud', downsample=50000, z_range=None, axis_range=None):
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
    if z_range is not None:
        zmin, zmax = z_range
        mask &= (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
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
        o3d.visualization.draw_geometries([pcd])
    else:
        if _HAS_PLOTLY:
            # Use Plotly scatter3d and save to standalone HTML (do not auto-open)
            rgb = (colors[:, ::-1]).astype(int)
            color_strs = [f'rgb({r},{g},{b})' for r, g, b in rgb]
            trace = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=1, color=color_strs, opacity=0.8)
            )
            fig = go.Figure(data=[trace])
            fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=title)
            out_html = Path('pointcloud.html').resolve()
            fig.write_html(str(out_html), auto_open=False)
            print(f'Point cloud saved to HTML: {out_html} — open this file in your browser.')
        else:
            print('Plotly is not installed. Install with `pip install plotly` to visualize the point cloud in the browser.')
            return


def parse_args():
    p = argparse.ArgumentParser(description='Estimate depth from camera-pair using calibration results and show 3D view')
    p.add_argument('--calib', default='calibration_results.npz', help='Calibration npz file')
    p.add_argument('--left', required=False, help='Left camera prefix (e.g., e00)')
    p.add_argument('--right', required=False, help='Right camera prefix (e.g., e01)')
    p.add_argument('--left-img', required=False, help='Left image path')
    p.add_argument('--right-img', required=False, help='Right image path')
    p.add_argument('--num-disp', type=int, default=192, help='Number of disparities for SGBM')
    p.add_argument('--block', type=int, default=5, help='Block size for SGBM')
    p.add_argument('--ply-out', default=None, help='Optional output PLY filename')
    p.add_argument('--downsample', type=int, default=100000, help='Max points to visualize')
    p.add_argument('--use-all', action='store_true', help='Use all calibrated cameras (find images in --images-dir)')
    p.add_argument('--images-dir', default='.', help='Root directory to search for per-camera images when using --use-all')
    p.add_argument('--ref-prefix', default=None, help='Reference camera prefix when using --use-all (defaults to first prefix in calib)')
    p.add_argument('--images-undistorted', action='store_true', help='Set if images in --images-dir are already undistorted (skips undistortion)')
    p.add_argument('--z-range', nargs=2, type=float, metavar=('ZMIN','ZMAX'), help='Display only points with Z between ZMIN and ZMAX')
    p.add_argument('--axis-range', nargs=6, type=float, metavar=('XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX'),
                   help='Specify xmin xmax ymin ymax zmin zmax as six space-separated floats to crop the point cloud before display')
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

    # Multi-view fusion mode
    if args.use_all:
        ref_pref = args.ref_prefix if args.ref_prefix is not None else (prefixes[0] if len(prefixes) > 0 else None)
        if ref_pref is None:
            print('No camera prefixes found in calibration')
            sys.exit(1)
        images_undist = args.images_undistorted or ('_undist' in args.images_dir)
        print(f'Using all cameras with reference: {ref_pref} (images dir: {args.images_dir}) undistorted={images_undist}')
        depth_map, ref_img, K_ref = compute_multi_view_depth(calib, args.images_dir, ref_pref, prefixes, num_disp=args.num_disp, block_size=args.block, images_undistorted=images_undist)
        pts, colors = depth_to_pointcloud_from_depthmap(depth_map, ref_img, K_ref)
        print(f'Generated {len(pts)} 3D points from multi-view fusion')
        if args.ply_out:
            save_ply(args.ply_out, pts, colors)
            print('Saved PLY to', args.ply_out)
        # parse axis-range if present. Accept six space-separated floats.
        axis_range_vals = None
        if args.axis_range:
            # argparse with nargs=6 will produce a list of 6 floats
            if isinstance(args.axis_range, (list, tuple)) and len(args.axis_range) == 6:
                axis_range_vals = [float(x) for x in args.axis_range]
            else:
                # legacy: allow comma-separated string if provided
                try:
                    vals = [float(x) for x in str(args.axis_range).split(',')]
                    if len(vals) == 6:
                        axis_range_vals = vals
                except Exception:
                    pass
        print(f'Visualizing with z_range={args.z_range} axis_range={axis_range_vals}')
        visualize_pointcloud(pts, colors, downsample=args.downsample, z_range=args.z_range, axis_range=axis_range_vals)
        return

    # Single-pair fallback (original behavior)
    K1, D1 = get_camera_matrices(calib, args.left)
    K2, D2 = get_camera_matrices(calib, args.right)
    images_undist_single = args.images_undistorted or ('_undist' in args.left_img) or ('_undist' in args.right_img)
    R, T = find_pair_pose(calib, args.left, args.right)
    if R is None or T is None:
        print('Relative pose for pair not found in calibration. Please check keys in', args.calib)
        sys.exit(1)

    left_img = cv2.imread(args.left_img)
    right_img = cv2.imread(args.right_img)
    if left_img is None or right_img is None:
        print('Failed to load images')
        sys.exit(1)

    # Resize images to calibration image size if provided
    key_size = f"{args.left}_image_size"
    if key_size in calib:
        sz = tuple(calib[key_size].astype(int).tolist())
        # calib image_size is (w,h)
        target = (sz[1], sz[0])
        if left_img.shape[:2] != target:
            left_img = cv2.resize(left_img, (target[1], target[0]))
        if right_img.shape[:2] != target:
            right_img = cv2.resize(right_img, (target[1], target[0]))

    rectL, rectR, Q = rectify_pair(left_img, right_img, K1, D1 if not images_undist_single else np.zeros_like(D1), K2, D2 if not images_undist_single else np.zeros_like(D2), R, T)

    rectL_gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    rectR_gray = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    disp = compute_disparity(rectL_gray, rectR_gray, min_disp=0, num_disp=args.num_disp, block_size=args.block)

    pts, colors = disparity_to_pointcloud(disp, Q, rectL)
    print(f'Generated {len(pts)} 3D points')

    if args.ply_out:
        save_ply(args.ply_out, pts, colors)
        print('Saved PLY to', args.ply_out)

    axis_range_vals = None
    if args.axis_range:
        if isinstance(args.axis_range, (list, tuple)) and len(args.axis_range) == 6:
            axis_range_vals = [float(x) for x in args.axis_range]
        else:
            try:
                vals = [float(x) for x in str(args.axis_range).split(',')]
                if len(vals) == 6:
                    axis_range_vals = vals
            except Exception:
                pass

    visualize_pointcloud(pts, colors, downsample=args.downsample, z_range=args.z_range, axis_range=axis_range_vals)


if __name__ == '__main__':
    main()
