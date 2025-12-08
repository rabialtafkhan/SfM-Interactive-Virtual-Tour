import cv2
import numpy as np


def detect_features(image_cv, use_sift=True):
    """
    Detect keypoints and descriptors in an image.
    """
    if use_sift:
        detector = cv2.SIFT_create()
    else:
        detector = cv2.ORB_create(nfeatures=5000)
    
    keypoints, descriptors = detector.detectAndCompute(image_cv, None)
    
    if descriptors is None:
        print("⚠️ No features detected in image")
        return keypoints, np.array([])
    
    return keypoints, descriptors


def match_features(des1, des2, method='flann', ratio_threshold=0.7):
    """
    Match features between two images using either FLANN or BFMatcher.
    """
    if method == 'flann':
        # FLANN uses kd-tree, optimized for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # Brute force matcher, good for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # KNN match to apply Lowe's ratio test
    matches = matcher.knnMatch(des1, des2, k=2)
    
    if matches is None or len(matches) == 0:
        print("⚠️ No matches found")
        return []
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


def extract_match_points(kp1, kp2, matches):
    """
    Extract 2D points from matched keypoints.
    """
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    return points1, points2
