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
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.032, aruco_dict)
    
    # Image list for calibration
    image_files = sorted(Path(image_dir).glob("*.png"))
    
    all_corners = []
    all_ids = []
    image_size = None
    
    # Detect ChArUco corners in each image
    for image_file in image_files:
        img = cv2.imread(str(image_file))
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            
            if charuco_corners is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
    
    if len(all_corners) == 0:
        print(f"No valid calibration images found for {camera_name}")
        return None
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None
    )
    
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


def calibrate_external_pose(calibration_data_list, image_dir):
    """
    Calibrate external pose (rotation and translation) between cameras.
    
    Args:
        calibration_data_list: List of calibration data for each camera
        image_dir: Base directory containing camera image folders
        
    Returns:
        Dictionary with relative poses between cameras
    """
    # Define ChArUco dictionary and board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.032, aruco_dict)
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    # Dictionary to store poses of each camera relative to the board
    camera_poses = {}
    
    for calib_data in calibration_data_list:
        camera_name = calib_data['camera_name']
        camera_matrix = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
        
        camera_dir = os.path.join(image_dir, camera_name.split('_')[0])
        image_files = sorted(Path(camera_dir).glob("*.png"))
        
        rvecs_list = []
        tvecs_list = []
        
        for image_file in image_files:
            img = cv2.imread(str(image_file))
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if len(corners) > 0:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                
                if charuco_corners is not None and len(charuco_corners) > 3:
                    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
                    )
                    
                    if ret:
                        rvecs_list.append(rvec)
                        tvecs_list.append(tvec)
        
        if len(rvecs_list) > 0:
            # Average the poses across all images
            avg_rvec = np.mean(rvecs_list, axis=0)
            avg_tvec = np.mean(tvecs_list, axis=0)
            
            camera_poses[camera_name] = {
                'rvec': avg_rvec,
                'tvec': avg_tvec
            }
            
            print(f"Calibrated external pose for {camera_name}")
    
    # Compute relative poses between cameras
    relative_poses = {}
    camera_names = list(camera_poses.keys())
    
    for i, camera1 in enumerate(camera_names):
        for j, camera2 in enumerate(camera_names):
            if i < j:
                # Compute relative pose from camera1 to camera2
                pair_name = f"{camera1}_to_{camera2}"
                
                # Convert rvec to rotation matrix
                R1, _ = cv2.Rodrigues(camera_poses[camera1]['rvec'])
                R2, _ = cv2.Rodrigues(camera_poses[camera2]['rvec'])
                
                # Relative rotation and translation
                R_rel = R2 @ R1.T
                rvec_rel, _ = cv2.Rodrigues(R_rel)
                tvec_rel = camera_poses[camera2]['tvec'] - R_rel @ camera_poses[camera1]['tvec']
                
                relative_poses[pair_name] = {
                    'rvec': rvec_rel,
                    'tvec': tvec_rel,
                    'rotation_matrix': R_rel
                }
    
    return relative_poses


def save_calibration_to_npz(calibration_data_list, relative_poses, output_path="calibration_results.npz"):
    """
    Save calibration results to npz format with camera names as keys.
    
    Args:
        calibration_data_list: List of calibration data dictionaries
        relative_poses: Dictionary with relative poses between cameras
        output_path: Path to save the npz file
    """
    save_dict = {}
    
    # Save single camera calibration data with camera name as key
    for calib_data in calibration_data_list:
        camera_name = calib_data['camera_name']
        # Extract just the prefix (e.g., 'e00' from 'e00_20260213_052739')
        camera_prefix = camera_name.split('_')[0] if '_' in camera_name else camera_name
        
        save_dict[f"{camera_prefix}_camera_matrix"] = calib_data['camera_matrix']
        save_dict[f"{camera_prefix}_dist_coeffs"] = calib_data['dist_coeffs']
        save_dict[f"{camera_prefix}_image_size"] = np.array(calib_data['image_size'])
        save_dict[f"{camera_prefix}_reprojection_error"] = calib_data['reprojection_error']
    
    # Save relative poses
    for pair_name, pose_data in relative_poses.items():
        save_dict[f"{pair_name}_rvec"] = pose_data['rvec']
        save_dict[f"{pair_name}_tvec"] = pose_data['tvec']
        save_dict[f"{pair_name}_rotation_matrix"] = pose_data['rotation_matrix']
    
    np.savez(output_path, **save_dict)
    print(f"\nCalibration results saved to: {output_path}")


def main():
    """Main calibration pipeline."""
    image_base_dir = "./collected_data/20260213_052739"
    
    # Get list of camera directories
    camera_dirs = [d for d in Path(image_base_dir).iterdir() if d.is_dir()]
    
    if len(camera_dirs) == 0:
        print(f"No camera directories found in {image_base_dir}")
        return
    
    # Calibrate each camera
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
    
    # Calibrate external poses
    print("\n" + "=" * 50)
    print("Starting external pose calibration...")
    print("=" * 50 + "\n")
    
    relative_poses = calibrate_external_pose(calibration_data_list, image_base_dir)
    
    # Print results
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
    
    # Save calibration results to npz
    save_calibration_to_npz(calibration_data_list, relative_poses, "calibration_results.npz")


if __name__ == "__main__":
    main()
