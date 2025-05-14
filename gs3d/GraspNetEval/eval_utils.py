import cv2
import numpy as np
import open3d as o3d


def reconstruct_3d_from_mask_and_depth(mask, depth_image, camera_matrix, workspace=None, inpainting=False):
    if inpainting:
        depth_image = cv2.inpaint(depth_image, (depth_image == 0).astype(np.uint8), 5, cv2.INPAINT_NS)

    if workspace is not None:
        (x1, y1, x2, y2) = workspace
        mask[:y1, :] = 0
        mask[y2:, :] = 0
        mask[:, :x1] = 0
        mask[:, x2:] = 0

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid_mask = mask > 0
    z = depth_image[valid_mask] / 1000.0
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    return np.vstack((x, y, z)).T

def reconstruct_3d_rgb_from_mask_and_rgb(mask, rgb_image, workspace=None):
    if workspace is not None:
        (x1, y1, x2, y2) = workspace
        mask[:y1, :] = 0
        mask[y2:, :] = 0
        mask[:, :x1] = 0
        mask[:, x2:] = 0

    valid_mask = mask > 0
    return rgb_image[valid_mask]


def get_masks(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    obj_masks = []

    for i in range(1, 256):
        # mask = (gray_image >= i - 1) & (gray_image < i)  
        mask = np.where(gray_image == i, 1, 0)
        if np.max(mask) > 0 and np.count_nonzero(mask) > 2500:
            obj_masks.append(mask)
    return obj_masks


def construct_pcd_from_depth(size, intrinsic_matrix, depth, colors, workspace):
    xmap, ymap = np.arange(size[1]), np.arange(size[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    (x1, y1, x2, y2) = (0, 0, 0, 0)
    if workspace is not None:
        (x1, y1, x2, y2) = workspace
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[
        1, 2]
    points_z = depth / 1000.0
    mask = (points_z > 0)
    points_x = (xmap + x1 - cx) / fx * points_z
    points_y = (ymap + y1 - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask]
    if colors is not None:
        colors = colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
