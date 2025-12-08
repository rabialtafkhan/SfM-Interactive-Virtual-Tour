import os
import glob
from PIL import Image
import numpy as np
import cv2


def load_images(image_dir, file_extension='*.jpg'):
    """
    Load all images from a directory.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, file_extension)))
    
    if len(image_paths) == 0:
        print(f"⚠️ No images found in {image_dir}")
        return [], []
    
    images_pil = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images_pil.append(img)
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")
            continue
    return images_pil, image_paths


def images_pil_to_cv(images_pil):
    """
    Convert PIL images to OpenCV format.
    """
    images_cv = []
    for img_pil in images_pil:
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        images_cv.append(img_cv)
    return images_cv


def get_image_dimensions(image_path):
    """
    Get image dimensions.
    """
    img = Image.open(image_path)
    return img.size


def load_image_pair(image_path_1, image_path_2):
    """
    Load a pair of images.
    """
    img1_pil = Image.open(image_path_1)
    img2_pil = Image.open(image_path_2)
    
    img1_cv = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2BGR)
    
    return img1_pil, img2_pil, img1_cv, img2_cv
