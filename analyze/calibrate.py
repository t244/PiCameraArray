import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from collections import defaultdict


def calibrate_single_camera(image_dir, camera_name):
    """
    Calibrate a single camera using ChArUco pattern.
    
    Args:
        image_dir: Directory containing calibration images
        camera_name: Name of the camera
        
    Returns:
        Dictionary with camera matrix, distortion coefficients, and calibration info
    """
    # Define ChArUco dictionary and board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((5, 7), 0.038, 0.0205, aruco_dict)
    
    # Image list for calibration
    image_files = sorted(Path(image_dir).glob("*.png"))
    
    all_corners = []
    all_ids = []
    image_size = None
    
    # Detect ChArUco corners in each image
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    for image_file in image_files:
        gray = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        image_size = gray.shape[::-1]

        corners, ids, rejected = detector.detectMarkers(gray)

        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )

            if charuco_corners is not None and len(charuco_corners) >= 12:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"  {image_file.name}: {len(corners)} markers, {len(charuco_corners)} charuco corners")
            else:
                n = len(charuco_corners) if charuco_corners is not None else 0
                print(f"  {image_file.name}: {len(corners)} markers, {n} charuco corners (skipped, need >= 12)")

    # Filter to keep only diverse views by subsampling
    if len(all_corners) > 30:
        step = len(all_corners) // 30
        all_corners = all_corners[::step]
        all_ids = all_ids[::step]
        print(f"Subsampled to {len(all_corners)} images for calibration diversity")

    if len(all_corners) < 3:
        print(f"Not enough valid calibration images for {camera_name} (found {len(all_corners)}, need at least 3)")
        return None

    print(f"Calibrating {camera_name} with {len(all_corners)} images...")

    # Calibrate camera
    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, image_size, None, None,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT
        )
    except cv2.error as e:
        print(f"Calibration failed for {camera_name}: {e}")
        return None
    
    if not ret:
        print(f"Calibration failed for {camera_name}")
        return None
    
    calibration_data = {
        'camera_name': camera_name,
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'image_size': image_size,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': ret
    }
    
    print(f"Calibrated {camera_name}: Reprojection error = {ret:.4f}")
    return calibration_data


def _extract_frame_number(filename):
    """Extract frame number from filename like 'e00_000041_20260217_061201_894.png'."""
    parts = filename.stem.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def calibrate_external_pose(calibration_data_list, image_dir):
    """
    Calibrate external pose (rotation and translation) between cameras.

    For each simultaneous frame, estimates each camera's board pose, then
    computes pairwise relative poses. The board position cancels out so the
    relative pose between cameras is consistent regardless of board movement.
    Per-frame relative poses are then averaged (using median for robustness).

    Args:
        calibration_data_list: List of calibration data for each camera
        image_dir: Base directory containing camera image folders

    Returns:
        (camera_poses, relative_poses) where:
            camera_poses: dict mapping camera prefix -> {'rvec', 'tvec', 'rotation_matrix'}
            relative_poses: dict mapping 'eXX_to_eYY' -> {'rvec', 'tvec', 'rotation_matrix'}
    """
    # Same dictionary and board as calibrate_single_camera
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((5, 7), 0.038, 0.0205, aruco_dict)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Step 1: Detect board pose per-frame per-camera
    # frame_poses[frame_number][camera_prefix] = (R, tvec)
    frame_poses = defaultdict(dict)
    calib_lookup = {}

    for calib_data in calibration_data_list:
        camera_name = calib_data['camera_name']
        camera_prefix = camera_name.split('_')[0] if '_' in camera_name else camera_name
        calib_lookup[camera_prefix] = calib_data

        camera_dir = os.path.join(image_dir, camera_prefix)
        if not os.path.isdir(camera_dir):
            camera_dir = os.path.join(image_dir, camera_name)
        image_files = sorted(Path(camera_dir).glob("*.png"))

        camera_matrix = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
        detected_count = 0

        for image_file in image_files:
            frame_num = _extract_frame_number(image_file)
            if frame_num is None:
                continue

            gray = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            corners, ids, rejected = detector.detectMarkers(gray)
            if len(corners) == 0:
                continue

            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if charuco_corners is None or len(charuco_corners) < 6:
                continue

            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
            )
            if ret:
                R, _ = cv2.Rodrigues(rvec)
                frame_poses[frame_num][camera_prefix] = (R, tvec)
                detected_count += 1

        print(f"  {camera_prefix}: detected board in {detected_count} frames")

    # Step 2: For each frame, compute pairwise relative poses
    # The relative pose between cameras is constant; board position cancels out.
    # Relative from cam_i to cam_j:  R_rel = R_j @ R_i^T,  t_rel = t_j - R_rel @ t_i
    pair_tvecs = defaultdict(list)
    pair_rmats = defaultdict(list)

    for frame_num, poses in frame_poses.items():
        cams_in_frame = sorted(poses.keys())
        if len(cams_in_frame) < 2:
            continue
        for i, cam1 in enumerate(cams_in_frame):
            for j, cam2 in enumerate(cams_in_frame):
                if i >= j:
                    continue
                R1, t1 = poses[cam1]
                R2, t2 = poses[cam2]
                R_rel = R2 @ R1.T
                t_rel = t2 - R_rel @ t1
                pair_key = f"{cam1}_to_{cam2}"
                pair_rmats[pair_key].append(R_rel)
                pair_tvecs[pair_key].append(t_rel.flatten())

    # Step 3: Average relative poses using median translation for robustness
    relative_poses = {}
    for pair_key in pair_tvecs:
        tvecs_arr = np.array(pair_tvecs[pair_key])
        rmats_arr = np.array(pair_rmats[pair_key])
        median_tvec = np.median(tvecs_arr, axis=0).reshape(3, 1)
        # Use mean rotation matrix (then re-orthogonalize via SVD)
        mean_R = np.mean(rmats_arr, axis=0)
        U, _, Vt = np.linalg.svd(mean_R)
        R_avg = U @ Vt
        if np.linalg.det(R_avg) < 0:
            R_avg = -R_avg
        rvec_avg, _ = cv2.Rodrigues(R_avg)

        relative_poses[pair_key] = {
            'rvec': rvec_avg,
            'tvec': median_tvec,
            'rotation_matrix': R_avg
        }
        print(f"  {pair_key}: {len(pair_tvecs[pair_key])} frames, "
              f"dist={np.linalg.norm(median_tvec)*1000:.1f} mm")

    # Step 4: Compute per-camera board pose from a single representative frame
    # Pick the frame with the most cameras detected
    best_frame = max(frame_poses.keys(), key=lambda f: len(frame_poses[f]))
    camera_poses = {}
    for cam_prefix, (R, tvec) in frame_poses[best_frame].items():
        rvec, _ = cv2.Rodrigues(R)
        camera_poses[cam_prefix] = {
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': R
        }
    print(f"\nUsing frame {best_frame} for board poses ({len(camera_poses)} cameras)")

    return camera_poses, relative_poses


def save_calibration_to_npz(calibration_data_list, camera_poses, relative_poses, output_path="calibration_results.npz"):
    """
    Save calibration results to npz format with camera names as keys.

    Args:
        calibration_data_list: List of calibration data dictionaries
        camera_poses: Dictionary mapping camera prefix -> board-relative pose
        relative_poses: Dictionary with relative poses between cameras
        output_path: Path to save the npz file
    """
    save_dict = {}

    # Save single camera calibration data with camera name as key
    for calib_data in calibration_data_list:
        camera_name = calib_data['camera_name']
        camera_prefix = camera_name.split('_')[0] if '_' in camera_name else camera_name

        save_dict[f"{camera_prefix}_camera_matrix"] = calib_data['camera_matrix']
        save_dict[f"{camera_prefix}_dist_coeffs"] = calib_data['dist_coeffs']
        save_dict[f"{camera_prefix}_image_size"] = np.array(calib_data['image_size'])
        save_dict[f"{camera_prefix}_reprojection_error"] = calib_data['reprojection_error']

    # Save per-camera board poses (absolute pose of each camera relative to the board)
    for cam_prefix, pose_data in camera_poses.items():
        save_dict[f"{cam_prefix}_board_rvec"] = pose_data['rvec']
        save_dict[f"{cam_prefix}_board_tvec"] = pose_data['tvec']
        save_dict[f"{cam_prefix}_board_rotation_matrix"] = pose_data['rotation_matrix']

    # Save relative poses between camera pairs
    for pair_name, pose_data in relative_poses.items():
        save_dict[f"{pair_name}_rvec"] = pose_data['rvec']
        save_dict[f"{pair_name}_tvec"] = pose_data['tvec']
        save_dict[f"{pair_name}_rotation_matrix"] = pose_data['rotation_matrix']

    # Save list of calibrated camera prefixes
    calibrated_cams = sorted(camera_poses.keys())
    save_dict["camera_list"] = np.array(calibrated_cams)

    np.savez(output_path, **save_dict)
    print(f"\nCalibration results saved to: {output_path}")


def plot_camera_poses_3d(camera_poses):
    """
    Plot all camera positions with labels and the ChArUco board in 3D.

    camera_poses maps camera prefix -> {'rvec', 'tvec', 'rotation_matrix'}.
    Each entry is the board pose as seen from that camera (OpenCV convention:
    X_camera = R * X_board + t), so the camera centre in board frame is -R^T @ t.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # ChArUco board dimensions (5 cols x 7 rows, square size 38 mm)
    board_w = 5 * 0.038  # metres
    board_h = 7 * 0.038

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- Board as a filled yellow quad at the world origin ---
    board_corners = np.array([
        [0,       0,       0],
        [board_w, 0,       0],
        [board_w, board_h, 0],
        [0,       board_h, 0],
    ])
    poly = Poly3DCollection([board_corners.tolist()],
                            alpha=0.35, facecolor='gold', edgecolor='darkorange', linewidth=1.5)
    ax.add_collection3d(poly)
    ax.text(board_w / 2, board_h / 2, 0, 'Board', ha='center', va='top',
            fontsize=9, color='darkorange', fontweight='bold')

    # --- Camera positions ---
    arrow_len = 0.015  # axis arrow length (15 mm)
    all_points = list(board_corners)  # collect every plotted point for equal-scale calc
    for cam_prefix, pose in sorted(camera_poses.items()):
        R = pose['rotation_matrix']
        t = pose['tvec'].flatten()

        # Camera centre in board (world) frame
        cam_pos = -R.T @ t
        all_points.append(cam_pos)

        # Camera axes expressed in world (board) frame
        x_dir = R.T @ np.array([1.0, 0.0, 0.0])
        y_dir = R.T @ np.array([0.0, 1.0, 0.0])
        z_dir = R.T @ np.array([0.0, 0.0, 1.0])

        ax.scatter(*cam_pos, color='steelblue', s=60, zorder=5, depthshade=False)
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2],
                f'  {cam_prefix}', fontsize=7, color='navy')

        # X axis – red, Y axis – green, Z axis (optical) – blue
        for direction, color in ((x_dir, 'red'), (y_dir, 'green'), (z_dir, 'blue')):
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                      direction[0] * arrow_len,
                      direction[1] * arrow_len,
                      direction[2] * arrow_len,
                      color=color, arrow_length_ratio=0.4, linewidth=1.2)

    # --- Equal axis scale (mpl 3-D does not support set_aspect('equal')) ---
    pts = np.array(all_points)
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 * 1.1  # 10 % margin
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Array – 3D Pose Relative to Calibration Board')

    plt.tight_layout()
    out_path = "camera_poses_3d.png"
    plt.savefig(out_path, dpi=150)
    print(f"3D pose plot saved to: {out_path}")
    plt.show()


def load_calibration_from_npz(npz_path):
    """
    Reload calibration results previously saved by save_calibration_to_npz.

    Returns:
        (calibration_data_list, camera_poses, relative_poses)
    """
    data = np.load(npz_path, allow_pickle=True)
    camera_list = list(data["camera_list"])

    calibration_data_list = []
    for cam in camera_list:
        entry = {'camera_name': cam}
        for field in ('camera_matrix', 'dist_coeffs', 'image_size', 'reprojection_error'):
            npz_key = f"{cam}_{field}"
            if npz_key in data:
                entry[field] = data[npz_key]
        calibration_data_list.append(entry)

    camera_poses = {}
    for cam in camera_list:
        if f"{cam}_board_rvec" in data:
            camera_poses[cam] = {
                'rvec': data[f"{cam}_board_rvec"],
                'tvec': data[f"{cam}_board_tvec"],
                'rotation_matrix': data[f"{cam}_board_rotation_matrix"],
            }

    relative_poses = {}
    for i, cam1 in enumerate(camera_list):
        for j, cam2 in enumerate(camera_list):
            if i >= j:
                continue
            pair_key = f"{cam1}_to_{cam2}"
            if f"{pair_key}_rvec" in data:
                relative_poses[pair_key] = {
                    'rvec': data[f"{pair_key}_rvec"],
                    'tvec': data[f"{pair_key}_tvec"],
                    'rotation_matrix': data[f"{pair_key}_rotation_matrix"],
                }

    print(f"Loaded calibration: {len(calibration_data_list)} cameras, "
          f"{len(camera_poses)} board poses, {len(relative_poses)} relative pairs")
    return calibration_data_list, camera_poses, relative_poses


def main():
    """Main calibration pipeline."""
    parser = argparse.ArgumentParser(description="Calibrate cameras using ChArUco pattern.")
    parser.add_argument("image_base_dir", nargs="?",
                        help="Base directory containing camera image folders "
                             "(required when running or forcing calibration)")
    parser.add_argument("--output", default="calibration_results.npz",
                        help="Path to save/load calibration npz (default: calibration_results.npz)")
    parser.add_argument("--recompute", action="store_true",
                        help="Force re-running calibration even if the npz file already exists")
    args = parser.parse_args()

    output_path = args.output

    # --- Load existing results or (re)compute ---
    if not args.recompute and Path(output_path).exists():
        print(f"Found existing calibration: {output_path}")
        print("Loading saved results (use --recompute to force recomputation).\n")
        calibration_data_list, camera_poses, relative_poses = load_calibration_from_npz(output_path)
    else:
        if args.image_base_dir is None:
            parser.error("image_base_dir is required when no saved calibration exists "
                         "or when --recompute is set")

        image_base_dir = args.image_base_dir
        camera_dirs = [d for d in Path(image_base_dir).iterdir() if d.is_dir()]

        if len(camera_dirs) == 0:
            print(f"No camera directories found in {image_base_dir}")
            return

        print("=" * 50)
        print("Starting single camera calibration...")
        print("=" * 50)

        calibration_data_list = []
        for camera_dir in sorted(camera_dirs):
            camera_name = camera_dir.name
            print(f"\nCalibrating camera: {camera_name}")
            calib_data = calibrate_single_camera(str(camera_dir), camera_name)
            if calib_data is not None:
                calibration_data_list.append(calib_data)

        if len(calibration_data_list) == 0:
            print("No cameras successfully calibrated")
            return

        print("\n" + "=" * 50)
        print("Starting external pose calibration...")
        print("=" * 50 + "\n")

        camera_poses, relative_poses = calibrate_external_pose(calibration_data_list, image_base_dir)

        save_calibration_to_npz(calibration_data_list, camera_poses, relative_poses, output_path)

    # --- Print results ---
    print("\n" + "=" * 50)
    print("Calibration Results")
    print("=" * 50)

    for camera_data in calibration_data_list:
        print(f"\n{camera_data['camera_name']}:")
        print(f"  Camera Matrix:\n{camera_data['camera_matrix']}")
        print(f"  Distortion Coefficients: {camera_data['dist_coeffs'].flatten()}")

    print("\nRelative Poses between Cameras:")
    for pair_name, pose_data in relative_poses.items():
        print(f"\n{pair_name}:")
        print(f"  Rotation Vector: {pose_data['rvec'].flatten()}")
        print(f"  Translation Vector: {pose_data['tvec'].flatten()}")

    # Plot 3D camera layout
    plot_camera_poses_3d(camera_poses)


if __name__ == "__main__":
    main()
