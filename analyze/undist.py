import numpy as np
import cv2
from pathlib import Path

# Load calibration results
calib_file = Path('calibration_results.npz')
calib_data = np.load(calib_file, allow_pickle=True)

# Source and destination directories
source_dir = Path('packed_data/000004')
dest_dir = Path('packed_data/000004_undist')

# Create destination directory
dest_dir.mkdir(parents=True, exist_ok=True)

# Process all image files in source directory
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
for image_file in sorted(source_dir.iterdir()):
    if image_file.suffix.lower() in image_extensions:
        # Extract camera name from filename (e.g., 'e00' from 'e00.png')
        camera_name = image_file.stem
        
        # Load calibration data for this specific camera
        try:
            K = calib_data[f'{camera_name}_camera_matrix']
            dist = calib_data[f'{camera_name}_dist_coeffs']
        except KeyError:
            print(f"Warning: Calibration data not found for {camera_name}, skipping")
            continue
        
        # Read image
        img = cv2.imread(str(image_file))
        
        if img is None:
            print(f"Failed to read {image_file}")
            continue
        
        # Undistort image
        h, w = img.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undist_img = cv2.undistort(img, K, dist, None, new_K)
        
        # Save undistorted image
        output_path = dest_dir / image_file.name
        cv2.imwrite(str(output_path), undist_img)
        print(f"Undistorted and saved: {output_path}")

print(f"All images undistorted and saved to {dest_dir}")
