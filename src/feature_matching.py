import cv2
import numpy as np

def match_features_flann(des1, des2, ratio_threshold=0.7):
    """
    Match features using FLANN (KD-tree).
    Best for SIFT descriptors.
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)
    
    if matches is None or len(matches) == 0:
        print("⚠️ No matches found (FLANN)")
        return []
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    print(f"FLANN: Found {len(good_matches)} good matches")
    return good_matches


def match_features_bf(des1, des2, ratio_threshold=0.7, use_hamming=True):
    """
    Match features using Brute Force matcher.
    Good for ORB descriptors.
    """
    if use_hamming:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = matcher.knnMatch(des1, des2, k=2)
    
    if matches is None or len(matches) == 0:
        print("⚠️ No matches found (BF)")
        return []
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    print(f"BF: Found {len(good_matches)} good matches")
    return good_matches


def draw_matches(image1, keypoints1, image2, keypoints2, matches, max_matches=50):
    """
    Draw matches between two images.
    """
    match_image = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return match_image


def extract_matched_points(keypoints1, keypoints2, matches):
    """
    Extract 2D point coordinates from matched keypoints.
    """
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    return points1, points2


def filter_matches_by_distance(matches, threshold=100.0):
    """
    Filter matches by distance threshold.
    """
    filtered = [m for m in matches if m.distance < threshold]
    print(f"Distance filter: {len(filtered)} matches (distance < {threshold})")
    return filtered

