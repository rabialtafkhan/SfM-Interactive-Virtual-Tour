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


def compute_intrinsic_matrix(image_width, image_height, focal_length=None):
    """
    Compute the camera intrinsic matrix K.
    
    As per project manual: assume fx = fy = image_width,
    and principal point (cx, cy) at image center.
    
    Args:
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        focal_length: Optional override for focal length (default: image_width)
    
    Returns:
        K: 3x3 intrinsic matrix
    """
    if focal_length is None:
        # Per project manual: focal length = image width
        focal_length = float(image_width)

    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K


def reconstruct_two_view(image_path_1, image_path_2, K=None, output_ply=None):
    """
    Phase 1: Two-view Structure from Motion pipeline.
    
    Implements all 10 steps from the course manual:
    1. Load two images
    2. Detect SIFT keypoints + descriptors
    3. FLANN matching with ratio test
    4. Extract matched points
    5. Compute Essential Matrix
    6. Recover pose (R, t)
    7. Build projection matrices
    8. Triangulate points
    9. Cheirality + depth filtering
    10. Save point cloud as .ply
    
    Args:
        image_path_1: Path to first image
        image_path_2: Path to second image
        K: Optional intrinsic matrix (computed if None)
        output_ply: Optional path to save PLY file
        
    Returns:
        points_3d: Nx3 array of 3D points
        colors: Nx3 array of RGB colors for each point
        R: Rotation matrix of camera 2 relative to camera 1
        t: Translation vector of camera 2 relative to camera 1
        K: Intrinsic matrix used
        pts1_inlier: 2D points in image 1 (inliers)
        pts2_inlier: 2D points in image 2 (inliers)
    """
    print("=" * 60)
    print("TWO-VIEW RECONSTRUCTION PIPELINE (Phase 1)")
    print("=" * 60)

    # Step 1: Load images
    print("\n[1/10] Loading images...")
    img1_pil, img2_pil, img1_cv, img2_cv = load_image_pair(image_path_1, image_path_2)
    h, w = img1_cv.shape[:2]
    print(f"    Image size: {w} x {h}")

    # Step 2: Create intrinsic matrix (if not provided)
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

    # Step 4: Match features using FLANN with ratio test
    print("\n[4/10] Matching features (FLANN + ratio test)...")
    good_matches = match_features_flann(des1, des2, ratio_threshold=0.7)

    if len(good_matches) < 8:
        print(f"    ERROR: Not enough matches ({len(good_matches)} < 8)")
        return None, None, None, None, K, None, None

    # Step 5: Extract matched point coordinates
    print("\n[5/10] Extracting matched points...")
    points1, points2 = extract_matched_points(kp1, kp2, good_matches)
    print(f"    Extracted {len(points1)} point pairs")

    # Step 6: Compute Essential Matrix with RANSAC
    print("\n[6/10] Computing Essential Matrix...")
    E, inlier_mask = compute_essential_matrix(points1, points2, K, threshold=1.0)

    if E is None:
        print("    ERROR: Failed to compute Essential Matrix")
        return None, None, None, None, K, None, None

    # Filter to inliers only
    pts1_inlier = points1[inlier_mask.ravel() == 1]
    pts2_inlier = points2[inlier_mask.ravel() == 1]
    print(f"    Inliers: {len(pts1_inlier)} / {len(points1)}")

    # Step 7: Recover camera pose (R, t)
    print("\n[7/10] Recovering camera pose (R, t)...")
    R, t, pose_mask = recover_pose_from_essential(E, pts1_inlier, pts2_inlier, K)

    if R is None:
        print("    ERROR: Failed to recover pose")
        return None, None, None, None, K, None, None

    # Filter points by pose mask (cheirality from recoverPose)
    pts1_valid = pts1_inlier[pose_mask.ravel() > 0]
    pts2_valid = pts2_inlier[pose_mask.ravel() > 0]

    # Step 8: Build projection matrices
    print("\n[8/10] Building projection matrices...")
    # P1 = K * [I | 0] for first camera at origin
    P1 = create_projection_matrix(K, np.eye(3), np.zeros((3, 1)))
    # P2 = K * [R | t] for second camera
    P2 = create_projection_matrix(K, R, t)
    print(f"    P1 shape: {P1.shape}")
    print(f"    P2 shape: {P2.shape}")

    # Step 9: Triangulate points
    print("\n[9/10] Triangulating 3D points...")
    points_3d = triangulate_points(P1, P2, pts1_valid, pts2_valid)
    print(f"    Triangulated {len(points_3d)} points")

    # Step 10: Filter by cheirality and depth
    print("\n[10/10] Filtering points (cheirality + depth)...")
    
    # Cheirality check: points must be in front of both cameras
    points_3d_filtered, valid_mask = filter_by_cheirality(points_3d, R, t)
    pts1_valid = pts1_valid[valid_mask]
    pts2_valid = pts2_valid[valid_mask]
    
    # Depth filtering: remove outliers
    points_3d_filtered, valid_mask = filter_by_depth(points_3d_filtered, depth_min=0.1, depth_max=500)
    pts1_valid = pts1_valid[valid_mask]
    pts2_valid = pts2_valid[valid_mask]
    
    # Reprojection error filtering
    points_3d_filtered, valid_mask = filter_by_reprojection_error(
        points_3d_filtered, pts1_valid, pts2_valid, P1, P2, threshold=2.0
    )
    pts1_valid = pts1_valid[valid_mask]
    pts2_valid = pts2_valid[valid_mask]

    # Extract colors from first image
    colors = []
    for pt in pts1_valid:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            bgr = img1_cv[y, x]
            colors.append([bgr[2], bgr[1], bgr[0]])  # BGR to RGB
        else:
            colors.append([128, 128, 128])
    colors = np.array(colors)

    # Save PLY if requested
    if output_ply and len(points_3d_filtered) > 0:
        from visualization import save_ply
        save_ply(output_ply, points_3d_filtered, colors)

    print("\n" + "=" * 60)
    print(f"RECONSTRUCTION COMPLETE: {len(points_3d_filtered)} points")
    print("=" * 60)

    return points_3d_filtered, colors, R, t, K, pts1_valid, pts2_valid


# ============================================================================
# PHASE 2: INCREMENTAL MULTI-VIEW SfM
# ============================================================================

class IncrementalSfM:
    """
    Phase 2: Incremental Structure from Motion pipeline.
    
    Implements:
    1. View-graph creation
    2. Sequential image registration
    3. Pose initialization using PnP
    4. Triangulation of new tracks
    5. Bundle-like refinement
    6. Persistent global point cloud
    """
    
    def __init__(self, K):
        """
        Initialize the incremental SfM pipeline.
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
        
        # Global state
        self.points_3d = np.empty((0, 3))  # Nx3 global point cloud
        self.point_colors = np.empty((0, 3))  # Nx3 RGB colors
        self.camera_poses = []  # List of (R, t) tuples
        self.camera_images = []  # List of image arrays
        
        # Feature tracking
        self.all_keypoints = []  # List of keypoints per image
        self.all_descriptors = []  # List of descriptors per image
        
        # Map from 3D point index to list of (image_idx, kp_idx)
        self.observations = {}
        
        print("IncrementalSfM initialized")
    
    def initialize_from_two_view(self, img1_cv, img2_cv):
        """
        Initialize the reconstruction from a two-view reconstruction.
        
        Args:
            img1_cv, img2_cv: OpenCV images (BGR)
            
        Returns:
            success: Boolean indicating if initialization succeeded
        """
        print("\n" + "=" * 60)
        print("INITIALIZING FROM TWO-VIEW RECONSTRUCTION")
        print("=" * 60)
        
        h, w = img1_cv.shape[:2]
        
        # Extract features
        print("\nExtracting features...")
        kp1, des1 = extract_sift_features(img1_cv)
        kp2, des2 = extract_sift_features(img2_cv)
        
        self.all_keypoints = [kp1, kp2]
        self.all_descriptors = [des1, des2]
        self.camera_images = [img1_cv, img2_cv]
        
        # Match features
        print("Matching features...")
        matches = match_features_flann(des1, des2, ratio_threshold=0.7)
        
        if len(matches) < 8:
            print(f"ERROR: Not enough matches ({len(matches)})")
            return False
        
        # Extract matched points
        pts1, pts2 = extract_matched_points(kp1, kp2, matches)
        
        # Compute Essential Matrix
        print("Computing Essential Matrix...")
        E, inlier_mask = compute_essential_matrix(pts1, pts2, self.K)
        
        if E is None:
            print("ERROR: Failed to compute E")
            return False
        
        pts1_inlier = pts1[inlier_mask.ravel() == 1]
        pts2_inlier = pts2[inlier_mask.ravel() == 1]
        matches_inlier = [m for i, m in enumerate(matches) if inlier_mask.ravel()[i] == 1]
        
        # Recover pose
        print("Recovering pose...")
        R, t, pose_mask = recover_pose_from_essential(E, pts1_inlier, pts2_inlier, self.K)
        
        if R is None:
            return False
        
        # Store camera poses
        self.camera_poses = [
            (np.eye(3), np.zeros((3, 1))),  # Camera 1 at origin
            (R, t.reshape(3, 1))  # Camera 2
        ]
        
        # Build projection matrices
        P1 = create_projection_matrix(self.K, np.eye(3), np.zeros((3, 1)))
        P2 = create_projection_matrix(self.K, R, t)
        
        # Filter by pose mask
        pts1_valid = pts1_inlier[pose_mask.ravel() > 0]
        pts2_valid = pts2_inlier[pose_mask.ravel() > 0]
        matches_valid = [m for i, m in enumerate(matches_inlier) if pose_mask.ravel()[i] > 0]
        
        # Triangulate
        print("Triangulating initial points...")
        points_3d = triangulate_points(P1, P2, pts1_valid, pts2_valid)
        
        # Filter points
        points_3d, valid_mask = filter_by_cheirality(points_3d, R, t)
        pts1_valid = pts1_valid[valid_mask]
        pts2_valid = pts2_valid[valid_mask]
        matches_valid = [m for i, m in enumerate(matches_valid) if valid_mask[i]]
        
        points_3d, valid_mask = filter_by_depth(points_3d, 0.1, 500)
        pts1_valid = pts1_valid[valid_mask]
        pts2_valid = pts2_valid[valid_mask]
        matches_valid = [m for i, m in enumerate(matches_valid) if valid_mask[i]]
        
        # Store global point cloud
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
        self.point_colors = np.array(colors)
        
        # Build observation map for tracking
        self.observations = {}
        for i, m in enumerate(matches_valid):
            self.observations[i] = [
                (0, m.queryIdx),  # Image 0, keypoint index
                (1, m.trainIdx)   # Image 1, keypoint index
            ]
        
        print(f"\nInitialization complete:")
        print(f"    Points: {len(self.points_3d)}")
        print(f"    Cameras: {len(self.camera_poses)}")
        
        return True
    
    def add_view(self, img_cv):
        """
        Add a new view to the reconstruction using PnP.
        
        Args:
            img_cv: OpenCV image (BGR)
            
        Returns:
            success: Boolean
        """
        new_img_idx = len(self.camera_images)
        prev_img_idx = new_img_idx - 1
        
        print(f"\n--- Adding view {new_img_idx} ---")
        
        h, w = img_cv.shape[:2]
        
        # Step 1: Extract features
        kp_new, des_new = extract_sift_features(img_cv)
        
        if len(kp_new) == 0:
            print("    ERROR: No features detected")
            return False
        
        # Step 2: Match with previous image
        des_prev = self.all_descriptors[prev_img_idx]
        kp_prev = self.all_keypoints[prev_img_idx]
        
        matches = match_features_flann(des_prev, des_new, ratio_threshold=0.75)
        
        if len(matches) < 10:
            print(f"    ERROR: Not enough matches ({len(matches)})")
            return False
        
        # Step 3: Find 2D-3D correspondences
        points_3d_for_pnp = []
        points_2d_for_pnp = []
        match_to_3d_idx = []
        
        for m in matches:
            prev_kp_idx = m.queryIdx
            new_kp_idx = m.trainIdx
            
            for pt3d_idx, obs_list in self.observations.items():
                for (obs_img_idx, obs_kp_idx) in obs_list:
                    if obs_img_idx == prev_img_idx and obs_kp_idx == prev_kp_idx:
                        points_3d_for_pnp.append(self.points_3d[pt3d_idx])
                        points_2d_for_pnp.append(kp_new[new_kp_idx].pt)
                        match_to_3d_idx.append((pt3d_idx, new_kp_idx))
                        break
        
        points_3d_for_pnp = np.array(points_3d_for_pnp, dtype=np.float64)
        points_2d_for_pnp = np.array(points_2d_for_pnp, dtype=np.float64)
        
        print(f"    Found {len(points_3d_for_pnp)} 2D-3D correspondences")
        
        if len(points_3d_for_pnp) < 6:
            print("    ERROR: Not enough 2D-3D correspondences for PnP")
            return False
        
        # Step 4: Solve PnP with RANSAC
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
            print("    ERROR: PnP failed")
            return False
        
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec.reshape(3, 1)
        
        print(f"    PnP succeeded with {len(inliers)} inliers")
        
        # Add observation for inlier points
        for idx in inliers.ravel():
            pt3d_idx, new_kp_idx = match_to_3d_idx[idx]
            self.observations[pt3d_idx].append((new_img_idx, new_kp_idx))
        
        # Store new camera
        self.camera_poses.append((R_new, t_new))
        self.camera_images.append(img_cv)
        self.all_keypoints.append(kp_new)
        self.all_descriptors.append(des_new)
        
        # Step 5: Triangulate new points
        P_prev = create_projection_matrix(
            self.K, 
            self.camera_poses[prev_img_idx][0],
            self.camera_poses[prev_img_idx][1]
        )
        P_new = create_projection_matrix(self.K, R_new, t_new)
        
        new_pts_prev = []
        new_pts_new = []
        new_kp_indices = []
        
        existing_kp_in_prev = set()
        for pt3d_idx, obs_list in self.observations.items():
            for (obs_img_idx, obs_kp_idx) in obs_list:
                if obs_img_idx == prev_img_idx:
                    existing_kp_in_prev.add(obs_kp_idx)
        
        for m in matches:
            if m.queryIdx not in existing_kp_in_prev:
                pt_prev = kp_prev[m.queryIdx].pt
                pt_new = kp_new[m.trainIdx].pt
                new_pts_prev.append(pt_prev)
                new_pts_new.append(pt_new)
                new_kp_indices.append((m.queryIdx, m.trainIdx))
        
        if len(new_pts_prev) > 0:
            new_pts_prev = np.array(new_pts_prev, dtype=np.float32)
            new_pts_new = np.array(new_pts_new, dtype=np.float32)
            
            new_points_3d = triangulate_points(P_prev, P_new, new_pts_prev, new_pts_new)
            
            new_points_3d, valid_mask = filter_by_cheirality(new_points_3d, R_new, t_new)
            new_pts_prev = new_pts_prev[valid_mask]
            new_pts_new = new_pts_new[valid_mask]
            new_kp_indices = [idx for i, idx in enumerate(new_kp_indices) if valid_mask[i]]
            
            new_points_3d, valid_mask = filter_by_depth(new_points_3d, 0.1, 500)
            new_pts_prev = new_pts_prev[valid_mask]
            new_pts_new = new_pts_new[valid_mask]
            new_kp_indices = [idx for i, idx in enumerate(new_kp_indices) if valid_mask[i]]
            
            start_idx = len(self.points_3d)
            self.points_3d = np.vstack([self.points_3d, new_points_3d])
            
            new_colors = []
            for pt in new_pts_new:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    bgr = img_cv[y, x]
                    new_colors.append([bgr[2], bgr[1], bgr[0]])
                else:
                    new_colors.append([128, 128, 128])
            self.point_colors = np.vstack([self.point_colors, np.array(new_colors)])
            
            for i, (prev_kp_idx, new_kp_idx) in enumerate(new_kp_indices):
                pt3d_idx = start_idx + i
                self.observations[pt3d_idx] = [
                    (prev_img_idx, prev_kp_idx),
                    (new_img_idx, new_kp_idx)
                ]
            
            print(f"    Triangulated {len(new_points_3d)} new points")
        
        print(f"    Total points: {len(self.points_3d)}")
        
        return True
    
    def refine_reconstruction(self, num_iterations=3):
        """
        Simple bundle-like refinement by filtering outliers.
        """
        print("\n" + "=" * 60)
        print("REFINING RECONSTRUCTION")
        print("=" * 60)
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            valid_points = np.ones(len(self.points_3d), dtype=bool)
            
            for pt3d_idx in range(len(self.points_3d)):
                if pt3d_idx not in self.observations:
                    continue
                    
                pt3d = self.points_3d[pt3d_idx]
                errors = []
                
                for (img_idx, kp_idx) in self.observations[pt3d_idx]:
                    if img_idx >= len(self.camera_poses):
                        continue
                        
                    R, t = self.camera_poses[img_idx]
                    P = create_projection_matrix(self.K, R, t)
                    
                    pt3d_h = np.append(pt3d, 1)
                    proj = P @ pt3d_h
                    proj = proj[:2] / proj[2]
                    
                    kp = self.all_keypoints[img_idx][kp_idx]
                    obs = np.array(kp.pt)
                    
                    error = np.linalg.norm(proj - obs)
                    errors.append(error)
                
                if len(errors) > 0:
                    avg_error = np.mean(errors)
                    if avg_error > 5.0:
                        valid_points[pt3d_idx] = False
            
            removed = np.sum(~valid_points)
            print(f"    Removed {removed} outliers (error > 5px)")
            
            if removed == 0:
                break
            
            old_to_new = {}
            new_idx = 0
            for old_idx in range(len(self.points_3d)):
                if valid_points[old_idx]:
                    old_to_new[old_idx] = new_idx
                    new_idx += 1
            
            self.points_3d = self.points_3d[valid_points]
            self.point_colors = self.point_colors[valid_points]
            
            new_observations = {}
            for old_idx, obs_list in self.observations.items():
                if old_idx in old_to_new:
                    new_observations[old_to_new[old_idx]] = obs_list
            self.observations = new_observations
        
        print(f"\nRefinement complete: {len(self.points_3d)} points remaining")
    
    def get_point_cloud(self):
        """Return the current point cloud and colors."""
        return self.points_3d, self.point_colors
    
    def get_camera_poses(self):
        """Return list of camera poses as (R, t) tuples."""
        return self.camera_poses
    
    def save_ply(self, filename):
        """Save the point cloud to a PLY file."""
        from visualization import save_ply
        save_ply(filename, self.points_3d, self.point_colors)
