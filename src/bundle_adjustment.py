import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def compute_reprojection_errors(points_3d, observations, camera_poses, K, keypoints_list):
    """
    Compute reprojection errors for all observations.
    
    Args:
        points_3d: Nx3 array of 3D points
        observations: dict mapping point_idx -> [(img_idx, kp_idx), ...]
        camera_poses: list of (R, t) tuples
        K: 3x3 intrinsic matrix
        keypoints_list: list of keypoints per image
        
    Returns:
        errors: list of (point_idx, img_idx, error) tuples
        mean_error: mean reprojection error
    """
    errors = []
    all_errors = []
    
    for pt_idx, obs_list in observations.items():
        if pt_idx >= len(points_3d):
            continue
            
        pt_3d = points_3d[pt_idx]
        
        for (img_idx, kp_idx) in obs_list:
            if img_idx >= len(camera_poses):
                continue
            if kp_idx >= len(keypoints_list[img_idx]):
                continue
                
            R, t = camera_poses[img_idx]
            t = t.reshape(3, 1)
            
            # Project 3D point to image
            Rt = np.hstack([R, t])
            P = K @ Rt
            
            pt_3d_h = np.append(pt_3d, 1)
            proj = P @ pt_3d_h
            
            if proj[2] <= 0:  # Behind camera
                continue
                
            proj_2d = proj[:2] / proj[2]
            
            # Get observed 2D point
            obs_2d = np.array(keypoints_list[img_idx][kp_idx].pt)
            
            # Compute error
            error = np.linalg.norm(proj_2d - obs_2d)
            errors.append((pt_idx, img_idx, error))
            all_errors.append(error)
    
    mean_error = np.mean(all_errors) if len(all_errors) > 0 else 0
    return errors, mean_error


def filter_high_error_points(points_3d, point_colors, observations, 
                              camera_poses, K, keypoints_list, 
                              threshold=5.0):
    """
    Remove 3D points with high reprojection error.
    
    Args:
        points_3d: Nx3 array of 3D points
        point_colors: Nx3 array of RGB colors
        observations: dict mapping point_idx -> [(img_idx, kp_idx), ...]
        camera_poses: list of (R, t) tuples
        K: 3x3 intrinsic matrix
        keypoints_list: list of keypoints per image
        threshold: maximum allowed reprojection error (pixels)
        
    Returns:
        filtered_points: Mx3 array
        filtered_colors: Mx3 array
        filtered_observations: updated dict
        removed_count: number of points removed
    """
    # Compute errors for each point
    point_errors = {}
    
    for pt_idx, obs_list in observations.items():
        if pt_idx >= len(points_3d):
            continue
            
        pt_3d = points_3d[pt_idx]
        errors = []
        
        for (img_idx, kp_idx) in obs_list:
            if img_idx >= len(camera_poses):
                continue
            if kp_idx >= len(keypoints_list[img_idx]):
                continue
                
            R, t = camera_poses[img_idx]
            t = t.reshape(3, 1)
            
            Rt = np.hstack([R, t])
            P = K @ Rt
            
            pt_3d_h = np.append(pt_3d, 1)
            proj = P @ pt_3d_h
            
            if proj[2] <= 0:
                errors.append(float('inf'))
                continue
                
            proj_2d = proj[:2] / proj[2]
            obs_2d = np.array(keypoints_list[img_idx][kp_idx].pt)
            
            error = np.linalg.norm(proj_2d - obs_2d)
            errors.append(error)
        
        if len(errors) > 0:
            point_errors[pt_idx] = np.mean(errors)
    
    # Find valid points
    valid_mask = np.ones(len(points_3d), dtype=bool)
    for pt_idx, avg_error in point_errors.items():
        if avg_error > threshold:
            valid_mask[pt_idx] = False
    
    # Create mapping from old to new indices
    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(points_3d)):
        if valid_mask[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    # Filter points and colors
    filtered_points = points_3d[valid_mask]
    filtered_colors = point_colors[valid_mask] if len(point_colors) == len(points_3d) else np.empty((0, 3))
    
    # Update observations
    filtered_observations = {}
    for old_idx, obs_list in observations.items():
        if old_idx in old_to_new:
            filtered_observations[old_to_new[old_idx]] = obs_list
    
    removed_count = len(points_3d) - len(filtered_points)
    
    return filtered_points, filtered_colors, filtered_observations, removed_count


def rodrigues_to_matrix(rvec):
    """Convert Rodrigues vector to rotation matrix."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def matrix_to_rodrigues(R):
    """Convert rotation matrix to Rodrigues vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def bundle_adjustment_residuals(params, n_cameras, n_points, camera_indices, 
                                 point_indices, points_2d, K):
    """
    Compute residuals for bundle adjustment.
    
    Args:
        params: flattened array of [camera_params..., point_params...]
                camera_params: 6 params per camera (rvec, tvec)
                point_params: 3 params per point (X, Y, Z)
        n_cameras: number of cameras
        n_points: number of 3D points
        camera_indices: array of camera index for each observation
        point_indices: array of point index for each observation
        points_2d: Nx2 array of observed 2D points
        K: 3x3 intrinsic matrix
        
    Returns:
        residuals: flattened array of reprojection errors (x and y)
    """
    # Extract camera parameters
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    # Compute projections
    residuals = []
    
    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        # Get camera pose
        rvec = camera_params[cam_idx, :3]
        tvec = camera_params[cam_idx, 3:6]
        
        # Get 3D point
        pt_3d = points_3d[pt_idx]
        
        # Project point
        R = rodrigues_to_matrix(rvec)
        pt_cam = R @ pt_3d + tvec
        
        if pt_cam[2] <= 0:
            # Point behind camera - large error
            residuals.extend([1000, 1000])
            continue
        
        # Project to image
        pt_proj = K @ pt_cam
        pt_proj = pt_proj[:2] / pt_proj[2]
        
        # Compute residual
        residuals.extend(pt_proj - points_2d[i])
    
    return np.array(residuals)


def run_bundle_adjustment(points_3d, observations, camera_poses, K, keypoints_list,
                          fix_first_camera=True, max_iterations=50):
    """
    Run full bundle adjustment to refine camera poses and 3D points.
    
    Args:
        points_3d: Nx3 array of 3D points
        observations: dict mapping point_idx -> [(img_idx, kp_idx), ...]
        camera_poses: list of (R, t) tuples
        K: 3x3 intrinsic matrix
        keypoints_list: list of keypoints per image
        fix_first_camera: if True, keep first camera at origin
        max_iterations: maximum optimization iterations
        
    Returns:
        refined_points: Nx3 array of refined 3D points
        refined_poses: list of refined (R, t) tuples
        final_error: mean reprojection error after optimization
    """
    n_cameras = len(camera_poses)
    n_points = len(points_3d)
    
    print(f"    Bundle Adjustment: {n_cameras} cameras, {n_points} points")
    
    # Build observation arrays
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for pt_idx, obs_list in observations.items():
        if pt_idx >= n_points:
            continue
        for (img_idx, kp_idx) in obs_list:
            if img_idx >= n_cameras:
                continue
            if kp_idx >= len(keypoints_list[img_idx]):
                continue
            
            camera_indices.append(img_idx)
            point_indices.append(pt_idx)
            points_2d.append(keypoints_list[img_idx][kp_idx].pt)
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    n_observations = len(camera_indices)
    print(f"    Total observations: {n_observations}")
    
    if n_observations < 10:
        print("    WARNING: Too few observations for bundle adjustment")
        return points_3d, camera_poses, 0
    
    # Initialize parameters
    camera_params = np.zeros((n_cameras, 6))
    for i, (R, t) in enumerate(camera_poses):
        camera_params[i, :3] = matrix_to_rodrigues(R)
        camera_params[i, 3:6] = t.flatten()
    
    # Flatten parameters
    x0 = np.hstack([camera_params.ravel(), points_3d.ravel()])
    
    # Compute initial error
    initial_residuals = bundle_adjustment_residuals(
        x0, n_cameras, n_points, camera_indices, point_indices, points_2d, K
    )
    initial_error = np.sqrt(np.mean(initial_residuals**2))
    print(f"    Initial RMS error: {initial_error:.3f} pixels")
    
    # Build sparsity matrix for efficiency
    m = n_observations * 2  # residuals
    n = n_cameras * 6 + n_points * 3  # parameters
    
    A = lil_matrix((m, n), dtype=int)
    
    for i in range(n_observations):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        
        # Camera parameters affect this observation
        for j in range(6):
            A[2*i, cam_idx*6 + j] = 1
            A[2*i + 1, cam_idx*6 + j] = 1
        
        # Point parameters affect this observation
        for j in range(3):
            A[2*i, n_cameras*6 + pt_idx*3 + j] = 1
            A[2*i + 1, n_cameras*6 + pt_idx*3 + j] = 1
    
    # Set bounds if fixing first camera
    if fix_first_camera:
        lower_bounds = -np.inf * np.ones_like(x0)
        upper_bounds = np.inf * np.ones_like(x0)
        eps = 1e-10
        lower_bounds[:6] = x0[:6] - eps  
        upper_bounds[:6] = x0[:6] + eps  
        bounds = (lower_bounds, upper_bounds)
    else:
        bounds = (-np.inf, np.inf)
    
    # Run optimization
    try:
        result = least_squares(
            bundle_adjustment_residuals,
            x0,
            jac_sparsity=A,
            verbose=0,
            x_scale='jac',
            ftol=1e-4,
            xtol=1e-4,
            method='trf',
            max_nfev=max_iterations,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
            bounds=bounds
        )
        
        # Extract results
        optimized_params = result.x
        camera_params = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
        refined_points = optimized_params[n_cameras * 6:].reshape((n_points, 3))
        
        # Convert camera params back to (R, t)
        refined_poses = []
        for i in range(n_cameras):
            R = rodrigues_to_matrix(camera_params[i, :3])
            t = camera_params[i, 3:6].reshape(3, 1)
            refined_poses.append((R, t))
        
        # Compute final error
        final_residuals = bundle_adjustment_residuals(
            optimized_params, n_cameras, n_points, 
            camera_indices, point_indices, points_2d, K
        )
        final_error = np.sqrt(np.mean(final_residuals**2))
        print(f"    Final RMS error: {final_error:.3f} pixels")
        print(f"    Improvement: {(1 - final_error/initial_error)*100:.1f}%")
        
        return refined_points, refined_poses, final_error
        
    except Exception as e:
        print(f"    Bundle adjustment failed: {e}")
        return points_3d, camera_poses, initial_error


def iterative_refinement(points_3d, point_colors, observations, camera_poses, 
                         K, keypoints_list, num_iterations=3, error_threshold=5.0):
    """
    Iteratively refine reconstruction by removing outliers and running BA.
    
    Args:
        points_3d: Nx3 array of 3D points
        point_colors: Nx3 array of RGB colors
        observations: dict mapping point_idx -> [(img_idx, kp_idx), ...]
        camera_poses: list of (R, t) tuples
        K: 3x3 intrinsic matrix
        keypoints_list: list of keypoints per image
        num_iterations: number of refinement iterations
        error_threshold: reprojection error threshold for outlier removal
        
    Returns:
        refined_points: refined 3D points
        refined_colors: corresponding colors
        refined_observations: updated observations
        refined_poses: refined camera poses
        stats: dict with refinement statistics
    """
    print("\n" + "=" * 60)
    print("ITERATIVE REFINEMENT")
    print("=" * 60)
    
    current_points = points_3d.copy()
    current_colors = point_colors.copy() if len(point_colors) > 0 else np.empty((0, 3))
    current_observations = observations.copy()
    current_poses = list(camera_poses)
    
    stats = {
        'initial_points': len(points_3d),
        'iterations': [],
        'final_points': 0,
        'final_error': 0
    }
    final_error = 0
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        print(f"    Points before: {len(current_points)}")
        
        # Compute current error
        _, mean_error = compute_reprojection_errors(
            current_points, current_observations, current_poses, K, keypoints_list
        )
        print(f"    Mean reprojection error: {mean_error:.3f} pixels")
        
        # Remove high-error points
        current_points, current_colors, current_observations, removed = filter_high_error_points(
            current_points, current_colors, current_observations,
            current_poses, K, keypoints_list, threshold=error_threshold
        )
        print(f"    Removed {removed} outliers")
        print(f"    Points after filtering: {len(current_points)}")
        
        if len(current_points) < 10:
            print("    WARNING: Too few points remaining!")
            # Compute error before breaking
            _, final_error = compute_reprojection_errors(
                current_points, current_observations, current_poses, K, keypoints_list
            )
            break
        
        # Run bundle adjustment (simplified - just recompute error)
        # Full BA is expensive, so we do it sparingly
        if iteration == num_iterations - 1 and len(current_points) >= 50:
            print("    Running bundle adjustment...")
            current_points, current_poses, final_error = run_bundle_adjustment(
                current_points, current_observations, current_poses,
                K, keypoints_list, fix_first_camera=True, max_iterations=30
            )
        else:
            _, final_error = compute_reprojection_errors(
                current_points, current_observations, current_poses, K, keypoints_list
            )
        
        stats['iterations'].append({
            'points': len(current_points),
            'removed': removed,
            'error': final_error
        })
        
        # Decrease threshold for next iteration
        error_threshold = max(2.0, error_threshold * 0.8)
    
    stats['final_points'] = len(current_points)
    stats['final_error'] = final_error
    
    print(f"\n--- Refinement Complete ---")
    print(f"    Initial points: {stats['initial_points']}")
    print(f"    Final points: {stats['final_points']}")
    print(f"    Final mean error: {stats['final_error']:.3f} pixels")
    
    return current_points, current_colors, current_observations, current_poses, stats



