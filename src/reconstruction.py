import cv2
import numpy as np
from load_data import load_image_pair
from preprocess import resize_image
from feature_extraction import extract_sift_features, extract_orb_features
from feature_matching import match_features_flann, extract_matched_points
from geometry import (compute_essential_matrix, recover_pose_from_essential, 
                      create_projection_matrix, triangulate_points, 
                      filter_by_cheirality, filter_by_depth)


def compute_intrinsic_matrix(image_width, image_height, focal_length=None):
    """
    Create camera intrinsic matrix K.
    """
    if focal_length is None:
        focal_length = image_width * 0.6
    
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def reconstruct_two_view(image_path_1, image_path_2, output_ply=None):
    """
    Full two-view reconstruction pipeline.
    """
    print("TWO-VIEW RECONSTRUCTION PIPELINE")
    print("\n[1/10] Loading images...")
    img1_pil, img2_pil, img1_cv, img2_cv = load_image_pair(image_path_1, image_path_2)
    h, w = img1_cv.shape[:2]
    
    print("\n[2/10] Creating intrinsic matrix...")
    K = compute_intrinsic_matrix(w, h)
    print(f"  K = \n{K}")
    
    print("\n[3/10] Extracting SIFT features...")
    kp1, des1 = extract_sift_features(img1_cv)
    kp2, des2 = extract_sift_features(img2_cv)
    
    if len(kp1) == 0 or len(kp2) == 0:
        print("⚠️ Failed to extract features")
        return None, None, None, K
    
    print("\n[4/10] Matching features...")
    good_matches = match_features_flann(des1, des2, ratio_threshold=0.7)
    
    if len(good_matches) < 8:
        print(f"⚠️ Not enough matches: {len(good_matches)}")
        return None, None, None, K
    
    print("\n[5/10] Extracting matched points...")
    points1, points2 = extract_matched_points(kp1, kp2, good_matches)
    
    print("\n[6/10] Computing essential matrix...")
    E, inlier_mask = compute_essential_matrix(points1, points2, K)
    
    if E is None:
        print("⚠️ Failed to compute essential matrix")
        return None, None, None, K
    
    print("\n[7/10] Recovering camera pose...")
    points1_inlier = points1[inlier_mask.ravel() == 1]
    points2_inlier = points2[inlier_mask.ravel() == 1]
    
    R, t, pose_mask = recover_pose_from_essential(E, points1_inlier, points2_inlier, K)
    
    if R is None:
        print("⚠️ Failed to recover pose")
        return None, None, None, K
    
    print("\n[8/10] Creating projection matrices...")
    P1 = create_projection_matrix(K, np.eye(3), np.zeros((3, 1)))
    P2 = create_projection_matrix(K, R, t)
    
    print("\n[9/10] Triangulating points...")
    points_3d = triangulate_points(P1, P2, points1_inlier, points2_inlier)
    
    print("\n[10/10] Filtering points...")
    points_3d = filter_by_cheirality(points_3d, pose_mask)
    points_3d = filter_by_depth(points_3d, depth_min=0.1, depth_max=1000)
    
    if output_ply:
        from visualization import save_ply
        save_ply(output_ply, points_3d)
    
    print("\n" + "="*60)
    print(f"✓ Reconstruction complete: {len(points_3d)} points")
    print("="*60 + "\n")
    
    return points_3d, R, t, K

