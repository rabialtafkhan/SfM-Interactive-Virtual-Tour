import cv2
import numpy as np

def compute_essential_matrix(points1, points2, K, threshold=1.0):
    """
    Compute the essential matrix using RANSAC.
    """
    if len(points1) < 8:
        print(f"⚠️ Not enough points for essential matrix: {len(points1)}")
        return None, None

    E, mask = cv2.findEssentialMat(
        points1, points2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold
    )

    if E is None:
        print("⚠️ Failed to compute essential matrix")
        return None, None
    print(f"Essential matrix computed ({np.sum(mask)} inliers)")
    return E, mask


def recover_pose_from_essential(E, points1, points2, K):
    """
    Recover camera pose (R, t) from an essential matrix.
    """
    num_inliers, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    print(f"Pose recovered ({num_inliers} points in front of both cameras)")
    return R, t, mask


def create_projection_matrix(K, R, t):
    """
    Create projection matrix P = K * [R | t].
    """
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt
    return P


def triangulate_points(P1, P2, points1, points2):
    """
    Triangulate points using linear least squares.
    """
    points1 = np.asarray(points1, dtype=np.float32)
    points2 = np.asarray(points2, dtype=np.float32)

    if points1.shape[0] != 2:
        points1 = points1.T
    if points2.shape[0] != 2:
        points2 = points2.T

    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d = (points_4d[:3] / points_4d[3]).T
    print(f"Triangulated {len(points_3d)} points")
    return points_3d


def filter_by_cheirality(points_3d, mask=None):
    """
    Filter points using cheirality check (points in front of cameras).
    """
    valid = points_3d[:, 2] > 0
    filtered = points_3d[valid]
    print(f"Cheirality filter: {len(filtered)} valid points out of {len(points_3d)}")
    return filtered


def filter_by_depth(points_3d, depth_min=0.1, depth_max=1000):
    """
    Filter 3D points by depth range.
    """
    depths = np.abs(points_3d[:, 2])
    valid = (depths >= depth_min) & (depths <= depth_max)
    filtered = points_3d[valid]
    removed = len(points_3d) - len(filtered)
    print(f"Depth filter: removed {removed} outliers, kept {len(filtered)} points")
    return filtered


