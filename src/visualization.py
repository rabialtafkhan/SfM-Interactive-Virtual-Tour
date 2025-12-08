import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_points(points_3d, title="3D Point Cloud", figsize=(10, 8)):
    """
    Plot 3D point cloud.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c='blue', marker='.', s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title} ({len(points_3d)} points)')
    
    plt.tight_layout()
    plt.show()


def plot_2d_projections(points_3d, title="2D Projections"):
    """
    Plot 2D projections of 3D points.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(points_3d[:, 0], points_3d[:, 1], s=1)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Projection')
    
    axes[1].scatter(points_3d[:, 0], points_3d[:, 2], s=1)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Projection')
    
    axes[2].scatter(points_3d[:, 1], points_3d[:, 2], s=1)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Projection')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_matches(image1, kp1, image2, kp2, matches, title="Feature Matches", max_matches=50):
    """
    Plot feature matches between two images.
    """
    match_img = cv2.drawMatches(
        image1, kp1,
        image2, kp2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} ({len(matches)} total matches)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_keypoints(image, keypoints, title="Detected Keypoints"):
    """
    Plot detected keypoints on image.
    """
    img_with_kp = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} ({len(keypoints)} features)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_image_pair(image1, image2, title="Image Pair"):
    """
    Plot two images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_ply(filename, points_3d, colors=None):
    """
    Save 3D points to PLY file.
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        if colors is None:
            for p in points_3d:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for i, p in enumerate(points_3d):
                c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    
    print(f"✓ Saved {len(points_3d)} points to {filename}")
