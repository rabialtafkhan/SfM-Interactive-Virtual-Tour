import cv2
import numpy as np

def resize_image(image, max_width=1920, max_height=1080):
    """
    Resize image to fit within max dimensions.
    """
    h, w = image.shape[:2]
    
    if w <= max_width and h <= max_height:
        return image
    
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)   
    print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    return resized


def equalize_histogram(image):
    """
    Apply histogram equalization to improve contrast.
    """
    if len(image.shape) == 3:  # color image
        # convert to HSV, equalize V channel, convert back
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge([h, s, v])
        equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:  # grayscale
        equalized = cv2.equalizeHist(image)
    
    return equalized


def normalize_image(image):
    """
    Normalize image to 0-1 range.
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized


def denoise_image(image, strength=10):
    """
    Apply denoising filter.
    """
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h=strength, hForColorComponents=strength, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)
    
    return denoised


def gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction for brightness adjustment.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    
    corrected = cv2.LUT(image, table)
    return corrected

