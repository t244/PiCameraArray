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
        """Load camera matrices and relative poses from calibration file."""
        try:
            calib_data = np.load(calib_file, allow_pickle=True)
            
            # Load camera matrices for each camera (e00-e15)
            for i in range(16):
                cam_name = f"e{i:02d}"
                try:
                    self.camera_matrices[cam_name] = calib_data[f'{cam_name}_camera_matrix']
                except KeyError:
                    pass
            
            # Load relative poses
            for key in calib_data.files:
                if '_to_' in key and '_rvec' in key:
                    pair_name = key.replace('_rvec', '')
                    try:
                        self.relative_poses[pair_name] = {
                            'rvec': calib_data[f'{pair_name}_rvec'],
                            'tvec': calib_data[f'{pair_name}_tvec'],
                            'rotation_matrix': calib_data[f'{pair_name}_rotation_matrix']
                        }
                    except KeyError:
                        pass
            
            print(f"Loaded calibration for {len(self.camera_matrices)} cameras")
            print(f"Loaded {len(self.relative_poses)} relative pose pairs")
            # derive baseline from relative poses
            try:
                self.compute_baselines()
            except Exception:
                # if baseline computation fails, keep defaults
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
            pair1 = f"{a}_to_{b}"
            pair2 = f"{b}_to_{a}"
            if pair1 in self.relative_poses and self.relative_poses[pair1].get('tvec') is not None:
                return np.linalg.norm(self.relative_poses[pair1]['tvec'])
            if pair2 in self.relative_poses and self.relative_poses[pair2].get('tvec') is not None:
                return np.linalg.norm(self.relative_poses[pair2]['tvec'])
            return None

        for i in range(16):
            row = i // 4
            col = i % 4
            a = f"e{i:02d}"
            if col < 3:
                j = i + 1
                b = f"e{j:02d}"
                d = try_distance(a, b)
                if d is not None:
                    hor.append(d)
            if row < 3:
                j = i + 4
                b = f"e{j:02d}"
                d = try_distance(a, b)
                if d is not None:
                    ver.append(d)

        all_d = hor + ver
        if all_d:
            # calibration tvecs are in meters; convert to mm
            mean_m = float(np.mean(all_d))
            self.baseline_mm = mean_m * 1000.0
        else:
            self.baseline_mm = 39.0
    
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
    
    def shift_image(self, img: np.ndarray, shift_x: float, shift_y: float):
        """Sub-pixel shift using affine transform."""
        h, w = img.shape[:2]
        M = np.float32([[1, 0, -shift_x],
                        [0, 1, -shift_y]])
        return cv2.warpAffine(img, M, (w, h), 
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    
    def synthesize(self, focus_depth: float, method: str = "mean"):
        """
        Create synthetic aperture image focused at specified depth.
        Uses calibration data for accurate geometric alignment.
        
        Args:
            focus_depth: Distance to focus plane (mm)
            method: 'mean', 'median', or 'trimmed_mean'
        """
        shifted = []
        
        for idx, img in enumerate(self.images):
            shift_x, shift_y = self.compute_shift_for_depth(idx, focus_depth)
            shifted.append(self.shift_image(img, shift_x, shift_y))
        
        stack = np.stack(shifted, axis=0).astype(np.float32)
        
        if method == "mean":
            result = np.mean(stack, axis=0)
        elif method == "median":
            result = np.median(stack, axis=0)
        elif method == "trimmed_mean":
            # Remove highest and lowest values, then average
            # This is more robust to occluders
            stack_sorted = np.sort(stack, axis=0)
            result = np.mean(stack_sorted[2:-2], axis=0)  # trim 2 from each end
        
        return result.astype(np.uint8)
    
    def create_focus_stack(self, depth_range: tuple, num_steps: int = 20):
        """Generate images focused at multiple depths."""
        depths = np.linspace(depth_range[0], depth_range[1], num_steps)
        return [(d, self.synthesize(d)) for d in depths]

if __name__ == "__main__":
    # Basic usage with calibration
    processor = SyntheticApertureProcessor(
        Path("packed_data/000004_undist/"),
        calib_file=Path("calibration_results.npz")
    )
    processor.load_images()
    
    # Create a focus sweep to find optimal depth
    for depth in range(200, 1000, 100):
        img = processor.synthesize(focus_depth=depth, method="mean")
        cv2.imwrite(f"focus_{depth}mm.png", img)
        