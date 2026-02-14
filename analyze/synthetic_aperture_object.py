"""
Align images based on a common object and synthesize a synthetic-aperture image.

Usage (example):
python synthetic_aperture_object.py --input-dir ../collected_data/20260213_153815 --pattern "e*/*.png" --ref 0 --method median --output out.png

Depends on: opencv-python, numpy, tqdm
"""

from pathlib import Path
import cv2
import numpy as np
import glob
import argparse
from tqdm import tqdm
import os


def load_images(paths):
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Can't read image: {p}")
        images.append(img)
    return images


def detect_and_describe(gray, method='ORB'):
    if method.upper() == 'ORB':
        detector = cv2.ORB_create(5000)
    elif method.upper() == 'AKAZE':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError('Unsupported detector: ' + method)
    kps, des = detector.detectAndCompute(gray, None)
    return kps, des


def match_descriptors(des1, des2, method='BF'):
    if des1 is None or des2 is None:
        return []
    # ORB/AKAZE descriptors are binary -> use Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
    except Exception:
        return []
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def compute_homography(kp1, kp2, matches, reproj_thresh=4.0):
    if len(matches) < 4:
        return None, None
    ptsA = np.float32([kp1[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
    return H, status


def align_images(images, ref_idx=0, detector='ORB', debug=False):
    """
    Align a list of BGR images to the reference image at ref_idx.
    Returns list of warped images and list of homographies (H mapping source -> ref).
    """
    ref_img = images[ref_idx]
    h_ref, w_ref = ref_img.shape[:2]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_kp, ref_des = detect_and_describe(ref_gray, method=detector)

    warped = []
    homographies = []

    for i, img in enumerate(images):
        if i == ref_idx:
            warped.append(img.copy())
            homographies.append(np.eye(3, dtype=np.float64))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = detect_and_describe(gray, method=detector)
        matches = match_descriptors(des, ref_des, method='BF')

        H, status = compute_homography(kp, ref_kp, matches)
        if H is None:
            # fallback: identity (no alignment)
            homographies.append(np.eye(3, dtype=np.float64))
            warped.append(img.copy())
            if debug:
                print(f"Image {i}: insufficient matches ({len(matches)}) - skipping warp")
            continue

        warped_img = cv2.warpPerspective(img, H, (w_ref, h_ref), flags=cv2.INTER_LINEAR)
        warped.append(warped_img)
        homographies.append(H)
    return warped, homographies


def synthesize_stack(images, method='median'):
    """
    Combine aligned images into a synthetic-aperture image.
    method: 'average', 'median', 'max', 'sum'
    Returns BGR uint8 image.
    """
    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    if method == 'average':
        out = np.mean(stack, axis=0)
    elif method == 'median':
        out = np.median(stack, axis=0)
    elif method == 'max':
        out = np.max(stack, axis=0)
    elif method == 'sum':
        out = np.sum(stack, axis=0)
    else:
        raise ValueError('Unknown method: ' + method)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def gather_image_paths(input_dir, pattern):
    # pattern may be a glob pattern; allow walking
    p = Path(input_dir)
    # Use glob with recursive patterns
    paths = sorted([Path(fp) for fp in glob.glob(str(p / pattern), recursive=True)])
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description='Align images on an object and synthesize synthetic aperture image')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory containing images')
    parser.add_argument('--pattern', '-p', default='*.png', help='Glob pattern to match images (relative to input dir)')
    parser.add_argument('--ref', '-r', type=int, default=0, help='Reference image index (0-based)')
    parser.add_argument('--detector', '-d', default='ORB', choices=['ORB', 'AKAZE'], help='Feature detector')
    parser.add_argument('--method', '-m', default='median', choices=['median','average','max','sum'], help='Combination method')
    parser.add_argument('--output', '-o', default='synthetic_aperture.png', help='Output filename')
    parser.add_argument('--limit', '-n', type=int, default=0, help='Limit number of images (0 = all)')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = gather_image_paths(args.input_dir, args.pattern)
    if len(paths) == 0:
        raise SystemExit(f'No images found for pattern: {args.pattern} in {args.input_dir}')
    if args.limit > 0:
        paths = paths[:args.limit]

    print(f'Found {len(paths)} images. Loading...')
    images = load_images(paths)
    print('Aligning images to reference index', args.ref)
    warped, Hs = align_images(images, ref_idx=args.ref, detector=args.detector, debug=args.debug)

    print('Synthesizing using method:', args.method)
    out = synthesize_stack(warped, method=args.method)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
