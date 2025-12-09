def add_view(self, img_cv):
    """
    Add a new view using PnP pose estimation.
    
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
    
    points_3d_for_pnp = np.array(points_3d_for_pnp, dtype=np.float64)
    points_2d_for_pnp = np.array(points_2d_for_pnp, dtype=np.float64)
    
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


