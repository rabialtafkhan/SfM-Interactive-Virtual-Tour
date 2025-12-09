import cv2
import numpy as np

from load_data import load_image_pair
from feature_extraction import extract_sift_features
from feature_matching import match_features_flann, extract_matched_points
from geometry import (
    compute_essential_matrix, recover_pose_from_essential,
    create_projection_matrix, triangulate_points,
    filter_by_cheirality, filter_by_depth, filter_by_reprojection_error
)


# =============================================================================
# INTRINSIC MATRIX
# =============================================================================

def compute_intrinsic_matrix(image_width, image_height, focal_length=None):
    """
    Compute the camera intrinsic matrix K.
    
    As per project manual: fx = fy = image_width, principal point at center.
    """
    if focal_length is None:
        focal_length = float(image_width)

    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K


# =============================================================================
# PHASE 1: TWO-VIEW RECONSTRUCTION
# =============================================================================

def reconstruct_two_view(image_path_1, image_path_2, K=None, output_ply=None):
    """
    Phase 1: Two-view Structure from Motion pipeline.
    
    Returns:
        points_3d, colors, R, t, K, pts1_valid, pts2_valid
    """
    print("=" * 60)
    print("TWO-VIEW RECONSTRUCTION PIPELINE (Phase 1)")
    print("=" * 60)

    # Step 1: Load images
    print("\n[1/10] Loading images...")
    img1_pil, img2_pil, img1_cv, img2_cv = load_image_pair(image_path_1, image_path_2)
    h, w = img1_cv.shape[:2]
    print(f"    Image size: {w} x {h}")

    # Step 2: Create intrinsic matrix
    print("\n[2/10] Creating intrinsic matrix...")
    if K is None:
        K = compute_intrinsic_matrix(w, h)
    print(f"    K =\n{K}")

    # Step 3: Extract SIFT features
    print("\n[3/10] Extracting SIFT features...")
    kp1, des1 = extract_sift_features(img1_cv)
    kp2, des2 = extract_sift_features(img2_cv)

    if len(kp1) == 0 or len(kp2) == 0:
        print("    ERROR: Failed to extract features")
        return None, None, None, None, K, None, None

    # Step 4: Match features
    print("\n[4/10] Matching features (FLANN + ratio test)...")
    good_matches = match_features_flann(des1, des2, ratio_threshold=0.7)

    if len(good_matches) < 8:
        print(f"    ERROR: Not enough matches ({len(good_matches)} < 8)")
        return None, None, None, None, K, None, None

    # Step 5: Extract matched points
    print("\n[5/10] Extracting matched points...")
    points1, points2 = extract_matched_points(kp1, kp2, good_matches)
    print(f"    Extracted {len(points1)} point pairs")

    # Step 6: Compute Essential Matrix
    print("\n[6/10] Computing Essential Matrix...")
    E, inlier_mask = compute_essential_matrix(points1, points2, K, threshold=1.0)

    if E is None:
        print("    ERROR: Failed to compute Essential Matrix")
        return None, None, None, None, K, None, None

    pts1_inlier = points1[inlier_mask.ravel() == 1]
    pts2_inlier = points2[inlier_mask.ravel() == 1]
    print(f"    Inliers: {len(pts1_inlier)} / {len(points1)}")

    # Step 7: Recover camera pose
    print("\n[7/10] Recovering camera pose (R, t)...")
    R, t, pose_mask = recover_pose_from_essential(E, pts1_inlier, pts2_inlier, K)

    if R is None:
        print("    ERROR: Failed to recover pose")
        return None, None, None, None, K, None, None

    pts1_valid = pts1_inlier[pose_mask.ravel() > 0]
    pts2_valid = pts2_inlier[pose_mask.ravel() > 0]

    # Step 8: Build projection matrices
    print("\n[8/10] Building projection matrices...")
    P1 = create_projection_matrix(K, np.eye(3), np.zeros((3, 1)))
    P2 = create_projection_matrix(K, R, t)

    # Step 9: Triangulate points
    print("\n[9/10] Triangulating 3D points...")
    points_3d = triangulate_points(P1, P2, pts1_valid, pts2_valid)

    # Step 10: Filter points
    print("\n[10/10] Filtering points...")
    points_3d_filtered, valid_mask = filter_by_cheirality(points_3d, R, t)
    pts1_valid = pts1_valid[valid_mask]
    pts2_valid = pts2_valid[valid_mask]
    
    points_3d_filtered, valid_mask = filter_by_depth(points_3d_filtered, 0.1, 500)
    pts1_valid = pts1_valid[valid_mask]
    pts2_valid = pts2_valid[valid_mask]
    
    points_3d_filtered, valid_mask = filter_by_reprojection_error(
        points_3d_filtered, pts1_valid, pts2_valid, P1, P2, threshold=2.0
    )
    pts1_valid = pts1_valid[valid_mask]
    pts2_valid = pts2_valid[valid_mask]

    # Extract colors
    colors = []
    for pt in pts1_valid:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            bgr = img1_cv[y, x]
            colors.append([bgr[2], bgr[1], bgr[0]])
        else:
            colors.append([128, 128, 128])
    colors = np.array(colors)

    if output_ply and len(points_3d_filtered) > 0:
        from visualization import save_ply
        save_ply(output_ply, points_3d_filtered, colors)

    print("\n" + "=" * 60)
    print(f"RECONSTRUCTION COMPLETE: {len(points_3d_filtered)} points")
    print("=" * 60)

    return points_3d_filtered, colors, R, t, K, pts1_valid, pts2_valid


# =============================================================================
# PHASE 2: INCREMENTAL MULTI-VIEW SfM
# =============================================================================

class IncrementalSfM:
    """
    Phase 2: Incremental Structure from Motion pipeline.
    
    Implements:
    1. Two-view initialization
    2. Sequential image registration via PnP
    3. Triangulation of new points
    4. Bundle adjustment refinement
    """
    
    def __init__(self, K):
        """Initialize with camera intrinsic matrix K."""
        self.K = K
        
        # Global reconstruction state
        self.points_3d = np.empty((0, 3))
        self.point_colors = np.empty((0, 3))
        self.camera_poses = []  # List of (R, t) tuples
        self.camera_images = []
        
        # Feature data per image
        self.all_keypoints = []
        self.all_descriptors = []
        
        # Observation tracking: point_idx -> [(img_idx, kp_idx), ...]
        self.observations = {}
        
        # Statistics
        self.stats = {
            'points_per_view': [],
            'inliers_per_view': [],
            'errors_per_view': []
        }
        
        print("IncrementalSfM initialized")
    
    def initialize_from_two_view(self, img1_cv, img2_cv):
        """
        Initialize reconstruction from two images.
        
        Args:
            img1_cv, img2_cv: OpenCV images (BGR)
            
        Returns:
            success: Boolean
        """
        print("\n" + "=" * 60)
        print("PHASE 2: INITIALIZING TWO-VIEW RECONSTRUCTION")
        print("=" * 60)
        
        h, w = img1_cv.shape[:2]
        
        # Extract features
        print("\n[1] Extracting features...")
        kp1, des1 = extract_sift_features(img1_cv)
        kp2, des2 = extract_sift_features(img2_cv)
        
        if len(kp1) == 0 or len(kp2) == 0:
            print("    ERROR: No features detected")
            return False
        
        self.all_keypoints = [kp1, kp2]
        self.all_descriptors = [des1, des2]
        self.camera_images = [img1_cv, img2_cv]
        
        # Match features
        print("\n[2] Matching features...")
        matches = match_features_flann(des1, des2, ratio_threshold=0.7)
        
        if len(matches) < 8:
            print(f"    ERROR: Not enough matches ({len(matches)})")
            return False
        
        pts1, pts2 = extract_matched_points(kp1, kp2, matches)
        
        # Compute Essential Matrix
        print("\n[3] Computing Essential Matrix...")
        E, inlier_mask = compute_essential_matrix(pts1, pts2, self.K)
        
        if E is None:
            return False
        
        pts1_inlier = pts1[inlier_mask.ravel() == 1]
        pts2_inlier = pts2[inlier_mask.ravel() == 1]
        matches_inlier = [m for i, m in enumerate(matches) if inlier_mask.ravel()[i] == 1]
        
        # Recover pose
        print("\n[4] Recovering camera pose...")
        R, t, pose_mask = recover_pose_from_essential(E, pts1_inlier, pts2_inlier, self.K)
        
        if R is None:
            return False
        
        # Store camera poses
        self.camera_poses = [
            (np.eye(3), np.zeros((3, 1))),
            (R, t.reshape(3, 1))
        ]
        
        # Build projection matrices
        P1 = create_projection_matrix(self.K, np.eye(3), np.zeros((3, 1)))
        P2 = create_projection_matrix(self.K, R, t)
        
        # Filter by pose mask
        pts1_valid = pts1_inlier[pose_mask.ravel() > 0]
        pts2_valid = pts2_inlier[pose_mask.ravel() > 0]
        matches_valid = [m for i, m in enumerate(matches_inlier) if pose_mask.ravel()[i] > 0]
        
        # Triangulate
        print("\n[5] Triangulating points...")
        points_3d = triangulate_points(P1, P2, pts1_valid, pts2_valid)
        
        # Filter
        points_3d, valid_mask = filter_by_cheirality(points_3d, R, t)
        pts1_valid = pts1_valid[valid_mask]
        pts2_valid = pts2_valid[valid_mask]
        matches_valid = [m for i, m in enumerate(matches_valid) if valid_mask[i]]
        
        points_3d, valid_mask = filter_by_depth(points_3d, 0.1, 500)
        pts1_valid = pts1_valid[valid_mask]
        pts2_valid = pts2_valid[valid_mask]
        matches_valid = [m for i, m in enumerate(matches_valid) if valid_mask[i]]
        
        self.points_3d = points_3d
        
        # Extract colors
        colors = []
        for pt in pts1_valid:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                bgr = img1_cv[y, x]
                colors.append([bgr[2], bgr[1], bgr[0]])
            else:
                colors.append([128, 128, 128])
        self.point_colors = np.array(colors) if colors else np.empty((0, 3))
        
        # Build observations
        self.observations = {}
        for i, m in enumerate(matches_valid):
            self.observations[i] = [
                (0, m.queryIdx),
                (1, m.trainIdx)
            ]
        
        self.stats['points_per_view'].extend([len(self.points_3d), len(self.points_3d)])
        
        print(f"\n--- Initialization Complete ---")
        print(f"    3D Points: {len(self.points_3d)}")
        print(f"    Cameras: {len(self.camera_poses)}")
        if len(self.points_3d) < 50:  
            print(f"    WARNING: Only {len(self.points_3d)} points - insufficient for incremental SfM")
            print("    Try a different image pair with better overlap")
            return False

        return True
    
    def add_view(self, img_cv):
        """
        Add a new view using PnP pose estimation.
        Searches ALL registered images for 2D-3D correspondences.
        
        Args:
            img_cv: OpenCV image (BGR)
            
        Returns:
            success: Boolean
        """
        new_img_idx = len(self.camera_images)
        
        print(f"\n--- Adding View {new_img_idx} ---")
        
        h, w = img_cv.shape[:2]
        
        # Step 1: Extract features
        print(f"    [1] Extracting features...")
        kp_new, des_new = extract_sift_features(img_cv)
        
        if len(kp_new) == 0:
            print("        ERROR: No features detected")
            return False
        
        # Step 2: Match with ALL registered images to find 2D-3D correspondences
        print(f"    [2] Finding 2D-3D correspondences from all views...")
        points_3d_for_pnp = []
        points_2d_for_pnp = []
        match_to_3d_idx = []
        used_3d_points = set()  # Avoid duplicates
        
        # Try matching with each registered image
        for img_idx in range(len(self.camera_images)):
            des_prev = self.all_descriptors[img_idx]
            kp_prev = self.all_keypoints[img_idx]
            
            matches = match_features_flann(des_prev, des_new, ratio_threshold=0.75)
            
            for m in matches:
                prev_kp_idx = m.queryIdx
                new_kp_idx = m.trainIdx
                
                # Look up if this keypoint corresponds to a 3D point
                for pt3d_idx, obs_list in self.observations.items():
                    if pt3d_idx in used_3d_points:
                        continue
                    for (obs_img_idx, obs_kp_idx) in obs_list:
                        if obs_img_idx == img_idx and obs_kp_idx == prev_kp_idx:
                            points_3d_for_pnp.append(self.points_3d[pt3d_idx])
                            points_2d_for_pnp.append(kp_new[new_kp_idx].pt)
                            match_to_3d_idx.append((pt3d_idx, new_kp_idx))
                            used_3d_points.add(pt3d_idx)
                            break
        
        points_3d_for_pnp = np.array(points_3d_for_pnp, dtype=np.float64) if points_3d_for_pnp else np.array([]).reshape(0, 3)
        points_2d_for_pnp = np.array(points_2d_for_pnp, dtype=np.float64) if points_2d_for_pnp else np.array([]).reshape(0, 2)
        
        print(f"        Found {len(points_3d_for_pnp)} 2D-3D correspondences")
        
        if len(points_3d_for_pnp) < 6:
            print("        ERROR: Not enough correspondences for PnP")
            return False
        
        # Step 3: Solve PnP with RANSAC
        print(f"    [3] Solving PnP...")
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d_for_pnp,
            points_2d_for_pnp,
            self.K,
            None,
            iterationsCount=1000,
            reprojectionError=3.0,
            confidence=0.99
        )
        
        if not success or inliers is None:
            print("        ERROR: PnP failed")
            return False
        
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec.reshape(3, 1)
        
        inlier_count = len(inliers)
        print(f"        PnP succeeded: {inlier_count} inliers")
        self.stats['inliers_per_view'].append(inlier_count)
        
        # Update observations for inlier points
        for idx in inliers.ravel():
            pt3d_idx, new_kp_idx = match_to_3d_idx[idx]
            self.observations[pt3d_idx].append((new_img_idx, new_kp_idx))
        
        # Store new camera
        self.camera_poses.append((R_new, t_new))
        self.camera_images.append(img_cv)
        self.all_keypoints.append(kp_new)
        self.all_descriptors.append(des_new)
        
        # Step 4: Triangulate new points with the best previous view
        print(f"    [4] Triangulating new points...")
        
        # Find best previous view (most matches)
        best_prev_idx = 0
        best_match_count = 0
        best_matches = []
        
        for img_idx in range(len(self.camera_images) - 1):
            des_prev = self.all_descriptors[img_idx]
            matches = match_features_flann(des_prev, des_new, ratio_threshold=0.75)
            if len(matches) > best_match_count:
                best_match_count = len(matches)
                best_prev_idx = img_idx
                best_matches = matches
        
        kp_prev = self.all_keypoints[best_prev_idx]
        
        P_prev = create_projection_matrix(
            self.K,
            self.camera_poses[best_prev_idx][0],
            self.camera_poses[best_prev_idx][1]
        )
        P_new = create_projection_matrix(self.K, R_new, t_new)
        
        # Find matches not yet triangulated
        existing_kp_in_prev = set()
        for pt3d_idx, obs_list in self.observations.items():
            for (obs_img_idx, obs_kp_idx) in obs_list:
                if obs_img_idx == best_prev_idx:
                    existing_kp_in_prev.add(obs_kp_idx)
        
        new_pts_prev = []
        new_pts_new = []
        new_kp_indices = []
        
        for m in best_matches:
            if m.queryIdx not in existing_kp_in_prev:
                new_pts_prev.append(kp_prev[m.queryIdx].pt)
                new_pts_new.append(kp_new[m.trainIdx].pt)
                new_kp_indices.append((m.queryIdx, m.trainIdx))
        
        if len(new_pts_prev) > 0:
            new_pts_prev = np.array(new_pts_prev, dtype=np.float32)
            new_pts_new = np.array(new_pts_new, dtype=np.float32)
            
            new_points_3d = triangulate_points(P_prev, P_new, new_pts_prev, new_pts_new)
            
            # Filter new points
            new_points_3d, valid_mask = filter_by_cheirality(new_points_3d, R_new, t_new)
            new_pts_prev = new_pts_prev[valid_mask]
            new_pts_new = new_pts_new[valid_mask]
            new_kp_indices = [idx for i, idx in enumerate(new_kp_indices) if valid_mask[i]]
            
            new_points_3d, valid_mask = filter_by_depth(new_points_3d, 0.1, 500)
            new_pts_prev = new_pts_prev[valid_mask]
            new_pts_new = new_pts_new[valid_mask]
            new_kp_indices = [idx for i, idx in enumerate(new_kp_indices) if valid_mask[i]]
            
            if len(new_points_3d) > 0:
                start_idx = len(self.points_3d)
                self.points_3d = np.vstack([self.points_3d, new_points_3d])
                
                # Extract colors
                new_colors = []
                for pt in new_pts_new:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        bgr = img_cv[y, x]
                        new_colors.append([bgr[2], bgr[1], bgr[0]])
                    else:
                        new_colors.append([128, 128, 128])
                
                if len(self.point_colors) > 0:
                    self.point_colors = np.vstack([self.point_colors, np.array(new_colors)])
                else:
                    self.point_colors = np.array(new_colors)
                
                # Update observations
                for i, (prev_kp_idx, new_kp_idx) in enumerate(new_kp_indices):
                    pt3d_idx = start_idx + i
                    self.observations[pt3d_idx] = [
                        (best_prev_idx, prev_kp_idx),
                        (new_img_idx, new_kp_idx)
                    ]
                
                print(f"        Added {len(new_points_3d)} new points")
        
        self.stats['points_per_view'].append(len(self.points_3d))
        print(f"    Total points: {len(self.points_3d)}")
        
        return True
    
    def refine_reconstruction(self, use_bundle_adjustment=True, num_iterations=3):
        """
        Refine the reconstruction using outlier removal and bundle adjustment.
        
        Args:
            use_bundle_adjustment: if True, run full BA
            num_iterations: number of refinement iterations
        """
        print("\n" + "=" * 60)
        print("REFINING RECONSTRUCTION")
        print("=" * 60)
        
        from bundle_adjustment import (
            iterative_refinement, 
            compute_reprojection_errors
        )
        
        # Compute initial error
        _, initial_error = compute_reprojection_errors(
            self.points_3d, self.observations, self.camera_poses,
            self.K, self.all_keypoints
        )
        print(f"\nInitial mean reprojection error: {initial_error:.3f} pixels")
        print(f"Initial point count: {len(self.points_3d)}")
        
        # Run iterative refinement
        (self.points_3d, self.point_colors, self.observations, 
         self.camera_poses, stats) = iterative_refinement(
            self.points_3d, self.point_colors, self.observations,
            self.camera_poses, self.K, self.all_keypoints,
            num_iterations=num_iterations,
            error_threshold=5.0
        )
        
        # Final error
        _, final_error = compute_reprojection_errors(
            self.points_3d, self.observations, self.camera_poses,
            self.K, self.all_keypoints
        )
        
        print(f"\n--- Refinement Summary ---")
        print(f"    Points: {stats['initial_points']} -> {stats['final_points']}")
        print(f"    Error: {initial_error:.3f} -> {final_error:.3f} pixels")
        
        self.stats['errors_per_view'].append(final_error)
    
    def get_point_cloud(self):
        """Return current point cloud and colors."""
        return self.points_3d, self.point_colors
    
    def get_camera_poses(self):
        """Return list of (R, t) camera poses."""
        return self.camera_poses
    
    def get_camera_centers(self):
        """Return camera center positions in world coordinates."""
        centers = []
        for R, t in self.camera_poses:
            C = -R.T @ t
            centers.append(C.flatten())
        return np.array(centers)
    
    def get_statistics(self):
        """Return reconstruction statistics."""
        return {
            'num_cameras': len(self.camera_poses),
            'num_points': len(self.points_3d),
            'points_per_view': self.stats['points_per_view'],
            'inliers_per_view': self.stats['inliers_per_view']
        }
    def find_best_initial_pair(self, images):
    """Find the image pair with best feature matches for initialization."""
    best_pair = (0, 1)
    best_inliers = 0
    
    for i in range(len(images)):
        for j in range(i+1, min(i+5, len(images))):  # Check nearby images
            kp1, des1 = extract_sift_features(images[i])
            kp2, des2 = extract_sift_features(images[j])
            matches = match_features_flann(des1, des2, ratio_threshold=0.7)
            
            if len(matches) > best_inliers:
                best_inliers = len(matches)
                best_pair = (i, j)
    
    return best_pair
    
    def save_ply(self, filename):
        """Save point cloud to PLY file."""
        from visualization import save_ply
        save_ply(filename, self.points_3d, self.point_colors)
    
    def save_cameras(self, filename):
        """Save camera poses to file."""
        camera_data = []
        for i, (R, t) in enumerate(self.camera_poses):
            camera_data.append({
                'index': i,
                'R': R.tolist(),
                't': t.flatten().tolist(),
                'center': (-R.T @ t).flatten().tolist()
            })
        
        import json
        with open(filename, 'w') as f:
            json.dump(camera_data, f, indent=2)
        print(f"Saved {len(camera_data)} camera poses to {filename}")


