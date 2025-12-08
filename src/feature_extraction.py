import cv2
import numpy as np

def extract_sift_features(image):
    """
    Extract SIFT features from image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None:
        print("⚠️ No SIFT features detected")
        return keypoints, np.array([])
    print(f"Detected {len(keypoints)} SIFT features")
    return keypoints, descriptors


def extract_orb_features(image, nfeatures=5000):
    """
    Extract ORB features from image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    if descriptors is None:
        print("⚠️ No ORB features detected")
        return keypoints, np.array([])
    print(f"Detected {len(keypoints)} ORB features")
    return keypoints, descriptors


def draw_keypoints(image, keypoints, radius=3, color=(0, 255, 0)):
    """
    Draw keypoints on image.
    """
    image_with_kp = cv2.drawKeypoints(image, keypoints, None, color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_with_kp


def filter_features_by_response(keypoints, descriptors, min_response=0.01):
    """
    Filter keypoints by response strength.
    """
    valid_idx = [i for i, kp in enumerate(keypoints) if kp.response >= min_response]
    
    filtered_kp = [keypoints[i] for i in valid_idx]
    filtered_des = descriptors[valid_idx] if len(valid_idx) > 0 else np.array([]) 
    print(f"Filtered to {len(filtered_kp)} features (response > {min_response})")
    return filtered_kp, filtered_des

