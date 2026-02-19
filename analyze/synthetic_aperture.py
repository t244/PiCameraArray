import argparse
import numpy as np
import cv2
from pathlib import Path

class SyntheticApertureProcessor:
    def __init__(self, image_dir: Path, calib_file: Path = None):
        self.image_dir = image_dir
        self.images = []
        self.image_names = []
        self.camera_matrices = {}
        self.relative_poses = {}
        self.grid_size = (4, 4)
        
        # Load calibration data if provided
        if calib_file and calib_file.exists():
            self.load_calibration(calib_file)
        else:
            print("Warning: No calibration file provided. Using default parameters.")
    
    def load_calibration(self, calib_file: Path):
        """Load camera matrices and board-relative poses from calibration file."""
        try:
            calib_data = np.load(calib_file, allow_pickle=True)

            # Load camera matrices for each camera (e00-e15)
            for i in range(16):
                cam_name = f"e{i:02d}"
                try:
                    self.camera_matrices[cam_name] = calib_data[f'{cam_name}_camera_matrix']
                except KeyError:
                    pass

            # Load per-camera board poses (absolute pose relative to board)
            self.board_poses = {}
            for i in range(16):
                cam_name = f"e{i:02d}"
                try:
                    self.board_poses[cam_name] = {
                        'rvec': calib_data[f'{cam_name}_board_rvec'],
                        'tvec': calib_data[f'{cam_name}_board_tvec'],
                        'rotation_matrix': calib_data[f'{cam_name}_board_rotation_matrix']
                    }
                except KeyError:
                    pass

            # Compute relative poses between all pairs from board poses
            self.relative_poses = {}
            cam_names = sorted(self.board_poses.keys())
            for i, cam1 in enumerate(cam_names):
                for j, cam2 in enumerate(cam_names):
                    if i != j:
                        R1 = self.board_poses[cam1]['rotation_matrix']
                        R2 = self.board_poses[cam2]['rotation_matrix']
                        t1 = self.board_poses[cam1]['tvec']
                        t2 = self.board_poses[cam2]['tvec']
                        R_rel = R2 @ R1.T
                        t_rel = t2 - R_rel @ t1
                        self.relative_poses[f"{cam1}_to_{cam2}"] = {
                            'rvec': cv2.Rodrigues(R_rel)[0],
                            'tvec': t_rel,
                            'rotation_matrix': R_rel
                        }

            print(f"Loaded calibration for {len(self.camera_matrices)} cameras")
            print(f"Loaded board poses for {len(self.board_poses)} cameras")
            print(f"Computed {len(self.relative_poses)} relative pose pairs")

            try:
                self.compute_baselines()
            except Exception:
                pass

        except Exception as e:
            print(f"Error loading calibration: {e}")

    def compute_baselines(self):
        """Compute average baseline (mm) between adjacent cameras using relative poses.

        It finds horizontal and vertical neighbor pairs (e.g. e00<->e01, e00<->e04)
        in the `relative_poses` dictionary and averages their Euclidean distances.
        If no pairs are available, sets a default baseline of 39.0 mm.
        """
        hor = []
        ver = []

        def try_distance(a, b):
            key = f"{a}_to_{b}"
            if key in self.relative_poses and self.relative_poses[key].get('tvec') is not None:
                return self.relative_poses[key]['tvec']
            return None

        print("\nNeighbor camera distances:")
        for i in range(16):
            row = i // 4
            col = i % 4
            a = f"e{i:02d}"
            if col < 3:
                j = i + 1
                b = f"e{j:02d}"
                tvec = try_distance(a, b)
                if tvec is not None:
                    d = np.linalg.norm(tvec)
                    hor.append(d)
                    t = tvec.flatten()
                    print(f"  {a}->{b} (horizontal): {d*1000:.1f} mm  (tx={t[0]*1000:.1f}, ty={t[1]*1000:.1f}, tz={t[2]*1000:.1f})")
            if row < 3:
                j = i + 4
                b = f"e{j:02d}"
                tvec = try_distance(a, b)
                if tvec is not None:
                    d = np.linalg.norm(tvec)
                    ver.append(d)
                    t = tvec.flatten()
                    print(f"  {a}->{b} (vertical):   {d*1000:.1f} mm  (tx={t[0]*1000:.1f}, ty={t[1]*1000:.1f}, tz={t[2]*1000:.1f})")

        all_d = hor + ver
        if all_d:
            mean_m = float(np.mean(all_d))
            self.baseline_mm = mean_m * 1000.0
            print(f"\nAverage horizontal baseline: {np.mean(hor)*1000:.1f} mm" if hor else "")
            print(f"Average vertical baseline:   {np.mean(ver)*1000:.1f} mm" if ver else "")
            print(f"Overall average baseline:    {self.baseline_mm:.1f} mm")
        else:
            self.baseline_mm = 39.0
            print("No neighbor pairs found, using default baseline: 39.0 mm")
    
    def get_focal_length(self, camera_name: str):
        """Extract focal length from camera matrix (in pixels)."""
        if camera_name in self.camera_matrices:
            K = self.camera_matrices[camera_name]
            # Focal length is the average of fx and fy from the camera matrix
            fx = K[0, 0]
            fy = K[1, 1]
            return (fx + fy) / 2.0
        else:
            # Default fallback for undistorted images (approximate)
            print(f"Warning: No camera matrix for {camera_name}. Using default focal length.")
            return 1800.0
        
    def load_images(self, pattern: str = "e*.png"):
        """Load all 16 images from the array."""
        paths = sorted(self.image_dir.glob(pattern))
        self.images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
        self.image_names = [p.stem for p in paths]  # e00, e01, ..., e15
        print(f"Loaded {len(self.images)} images")
        return self
    
    
    def compute_shift_for_depth(self, cam_index: int, focus_depth: float):
        """
        Calculate pixel shift to align a specific depth plane using calibration data.
        
        Args:
            cam_index: Index of camera (0-15)
            focus_depth: Distance to focus plane (mm)
            baseline: Default camera spacing (39mm for your array)
        
        Returns:
            (shift_x, shift_y) in pixels
        """
        # Get camera name (e00-e15)
        camera_name = self.image_names[cam_index] if cam_index < len(self.image_names) else f"e{cam_index:02d}"
        
        # Get focal length from calibration or use default
        focal_length_px = self.get_focal_length(camera_name)
        
        # Reference camera at center (1.5, 1.5 in 4x4 grid)
        center = 1.5
        cam_row = cam_index // 4
        cam_col = cam_index % 4
        
        # Determine baseline in mm from calibration if available
        baseline_mm = getattr(self, 'baseline_mm', 39.0)

        # Physical offset from center
        offset_x = (cam_col - center) * baseline_mm  # mm
        offset_y = (cam_row - center) * baseline_mm  # mm
        
        # Compute shift using perspective principles:
        # disparity = baseline * focal_length / depth
        # shift_pixels = baseline * focal_length / depth
        shift_x = offset_x * focal_length_px / focus_depth
        shift_y = offset_y * focal_length_px / focus_depth
        
        return shift_x, shift_y

    def get_relative_RT(self, cam_from: str, cam_to: str):
        """Return relative rotation R and translation t (meters) mapping X_from -> X_to.

        All pairs are precomputed from board poses during load_calibration.
        Returns (R, t) or (None, None) if unavailable.
        """
        key = f"{cam_from}_to_{cam_to}"
        if key in self.relative_poses:
            entry = self.relative_poses[key]
            return entry.get('rotation_matrix'), entry.get('tvec')
        return None, None

    def compute_homography_to_center(self, cam_from: str, cam_to: str, depth_mm: float):
        """Compute homography mapping pixels from cam_from to cam_to for a plane at depth_mm.

        Uses relative transform X_to = R X_from + t (t in meters) and intrinsics K.
        Assumes plane is fronto-parallel to cam_to (normal [0,0,1]) and depth measured
        along cam_to z-axis. depth_mm is converted to meters.
        Returns 3x3 homography or None if insufficient data.
        """
        R, t = self.get_relative_RT(cam_from, cam_to)
        if R is None or t is None:
            return None

        K_from = self.camera_matrices.get(cam_from)
        K_to = self.camera_matrices.get(cam_to)
        if K_from is None or K_to is None:
            return None

        # plane normal in cam_to frame (frontal)
        n = np.array([[0.0], [0.0], [1.0]])
        d = float(depth_mm) / 1000.0  # convert mm to meters

        # Homography: H = K_to * (R - (t * n^T)/d) * K_from^{-1}
        H = K_to @ (R - (t.reshape(3,1) @ n.T) / d) @ np.linalg.inv(K_from)
        # Normalize
        H = H / H[2,2]
        return H

    def warp_all_to_center(self, focus_depth: float, center_cam: str = None, log_dir: Path = None):
        """Warp all loaded images into the center view using homographies.

        Args:
            focus_depth: plane depth in mm used to compute homographies
            center_cam: camera name used as the reference (e.g., 'e05'); if None,
                        uses the first loaded image as center.
            log_dir: if provided, each warped image is saved here as
                     {cam_name}_{depth}mm.png for debugging.
        Returns:
            list of warped images (same order as self.images). If homography
            cannot be computed for a camera, the original image is returned.
        """
        if len(self.images) == 0:
            return []

        if center_cam is None:
            # Geometric center of the grid: upper-left of the 4 inner cameras.
            # For a 4x4 grid this is row 1, col 1  →  index 5  (e05).
            center_cam = self.image_names[(self.grid_size[0] // 2 - 1) * self.grid_size[1] + (self.grid_size[1] // 2 - 1)] if self.image_names else 'e05'

        # output size from center image
        center_idx = None
        if center_cam in self.image_names:
            center_idx = self.image_names.index(center_cam)
        else:
            center_idx = len(self.image_names) // 2

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        h, w = self.images[center_idx].shape[:2]
        warped = []
        for idx, img in enumerate(self.images):
            cam_from = self.image_names[idx]
            H = self.compute_homography_to_center(cam_from, center_cam, focus_depth)
            if H is None:
                # fallback: no homography, use identity
                warped_img = img
            else:
                warped_img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                
                # approximate pixel shift: the homography maps (0,0,1) to H*(0,0,1)
                # translation is roughly the last column of H (before normalization) relative to identity
                shift_x_approx = H[0, 2]
                shift_y_approx = H[1, 2]
                print(f"  {cam_from}: shift ≈ ({shift_x_approx:.1f}, {shift_y_approx:.1f}) px")
            warped.append(warped_img)

            if log_dir is not None:
                log_path = log_dir / f"{cam_from}_{int(focus_depth)}mm.png"
                cv2.imwrite(str(log_path), warped_img)

        return warped
    
    def shift_image(self, img: np.ndarray, shift_x: float, shift_y: float):
        """Sub-pixel shift using affine transform."""
        h, w = img.shape[:2]
        M = np.float32([[1, 0, -shift_x],
                        [0, 1, -shift_y]])
        return cv2.warpAffine(img, M, (w, h), 
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    
    def synthesize(self, focus_depth: float, method: str = "mean", center_cam: str = None, log_dir: Path = None):
        """
        Create synthetic aperture image focused at specified depth.
        Uses homography warping when calibration data is available,
        falls back to grid-based shift otherwise.

        Args:
            focus_depth: Distance to focus plane (mm)
            method: 'mean', 'median', or 'trimmed_mean'
            center_cam: Reference camera name (e.g. 'e05')
            log_dir: if provided, warped images are saved here for debugging
        """
        # Try homography-based warping first (uses actual calibration poses)
        if self.relative_poses and self.camera_matrices:
            aligned = self.warp_all_to_center(focus_depth, center_cam, log_dir=log_dir)
        else:
            # Fallback: shift-based alignment assuming regular grid
            aligned = []
            for idx, img in enumerate(self.images):
                shift_x, shift_y = self.compute_shift_for_depth(idx, focus_depth)
                aligned.append(self.shift_image(img, shift_x, shift_y))

        stack = np.stack(aligned, axis=0).astype(np.float32)

        if method == "mean":
            result = np.mean(stack, axis=0)
        elif method == "median":
            result = np.median(stack, axis=0)
        elif method == "trimmed_mean":
            # Remove highest and lowest values, then average
            # This is more robust to occluders
            stack_sorted = np.sort(stack, axis=0)
            trim = min(2, len(aligned) // 4)
            if trim > 0 and len(aligned) > 2 * trim:
                result = np.mean(stack_sorted[trim:-trim], axis=0)
            else:
                result = np.mean(stack, axis=0)

        return result.astype(np.uint8)
    
    def create_focus_stack(self, depth_range: tuple, num_steps: int = 20):
        """Generate images focused at multiple depths."""
        depths = np.linspace(depth_range[0], depth_range[1], num_steps)
        return [(d, self.synthesize(d)) for d in depths]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic aperture processing from camera array images.")
    parser.add_argument("image_dir", help="Directory containing undistorted camera images")
    parser.add_argument("--calib", default="calibration_results.npz",
                        help="Path to calibration npz file (default: calibration_results.npz)")
    args = parser.parse_args()

    processor = SyntheticApertureProcessor(
        Path(args.image_dir),
        calib_file=Path(args.calib)
    )
    processor.load_images()

    # Create a focus sweep to find optimal depth
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "log"
    for depth in range(300, 1000, 100):
        img = processor.synthesize(focus_depth=depth, method="trimmed_mean", log_dir=log_dir)
        cv2.imwrite(str(output_dir / f"focus_{depth}mm.png"), img)
        