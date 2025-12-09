import cv2
import numpy as np


def compute_essential_matrix(points1, points2, K, threshold=1.0):
    """
    Compute the Essential Matrix using RANSAC.
    """
    if len(points1) < 8:
        print(f"    WARNING: Not enough points for essential matrix: {len(points1)}")
        return None, None
    
    E, mask = cv2.findEssentialMat(
        points1, points2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold
    )
    
    if E is None:
        print("    WARNING: Failed to compute essential matrix")
        return None, None
    
    inlier_count = np.sum(mask)
    print(f"    Essential matrix computed ({inlier_count} inliers / {len(points1)} points)")
    
    return E, mask


def recover_pose_from_essential(E, points1, points2, K):
    """
    Recover camera pose (R, t) from Essential Matrix.
    """
    num_inliers, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    print(f"    Pose recovered ({num_inliers} points in front of both cameras)")
    return R, t, mask


def create_projection_matrix(K, R, t):
    """
    Create a 3x4 projection matrix P = K * [R | t].
    """
    t = t.reshape(3, 1)
    Rt = np.hstack([R, t])
    P = K @ Rt
    return P


def triangulate_points(P1, P2, points1, points2):
    """
    Triangulate 3D points from 2D correspondences.
    """
    points1 = np.asarray(points1, dtype=np.float32)
    points2 = np.asarray(points2, dtype=np.float32)
    
    if points1.shape[0] != 2:
        points1 = points1.T
    if points2.shape[0] != 2:
        points2 = points2.T
    
    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    print(f"    Triangulated {len(points_3d)} points")
    return points_3d


def filter_by_cheirality(points_3d, R=None, t=None):
    """
    Filter points by cheirality check (must be in front of both cameras).
    
    Returns:
        filtered_points: Mx3 array of valid points
        valid_mask: N-element boolean mask
    """
    valid_cam1 = points_3d[:, 2] > 0
    
    if R is not None and t is not None:
        t = t.reshape(3, 1)
        points_cam2 = (R @ points_3d.T + t).T
        valid_cam2 = points_cam2[:, 2] > 0
        valid_mask = valid_cam1 & valid_cam2
    else:
        valid_mask = valid_cam1
    
    filtered_points = points_3d[valid_mask]
    removed = len(points_3d) - len(filtered_points)
    print(f"    Cheirality filter: kept {len(filtered_points)}, removed {removed}")
    
    return filtered_points, valid_mask


def filter_by_depth(points_3d, depth_min=0.1, depth_max=500):
    """
    Filter 3D points by depth range to remove outliers.
    
    Returns:
        filtered_points: Mx3 array of valid points
        valid_mask: N-element boolean mask
    """
    depths = np.abs(points_3d[:, 2])
    valid_mask = (depths >= depth_min) & (depths <= depth_max)
    
    filtered_points = points_3d[valid_mask]
    removed = len(points_3d) - len(filtered_points)
    print(f"    Depth filter [{depth_min}, {depth_max}]: kept {len(filtered_points)}, removed {removed}")
    
    return filtered_points, valid_mask


def filter_by_reprojection_error(points_3d, points1, points2, P1, P2, threshold=2.0):
    """
    Filter 3D points by reprojection error.
    
    Returns:
        filtered_points: Mx3 array of valid points
        valid_mask: N-element boolean mask
    """
    n_points = len(points_3d)
    valid_mask = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        pt_3d = points_3d[i]
        pt_3d_h = np.append(pt_3d, 1)
        
        proj1 = P1 @ pt_3d_h
        proj1 = proj1[:2] / proj1[2]
        error1 = np.linalg.norm(proj1 - points1[i])
        
        proj2 = P2 @ pt_3d_h
        proj2 = proj2[:2] / proj2[2]
        error2 = np.linalg.norm(proj2 - points2[i])
        
        if error1 > threshold or error2 > threshold:
            valid_mask[i] = False
    
    filtered_points = points_3d[valid_mask]
    removed = n_points - len(filtered_points)
    print(f"    Reprojection filter (threshold={threshold}px): kept {len(filtered_points)}, removed {removed}")
    
    return filtered_points, valid_mask
