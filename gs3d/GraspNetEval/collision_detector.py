""" Collision detection to remove collided grasp pose predictions.
Author: chenxi-wang
"""

import numpy as np
import open3d as o3d


class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """

    def __init__(self, scene_points):
        self.finger_width = 0.02
        self.depth_base = 0.012
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, return_empty_grasp=False, empty_thresh=0.01):
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:, np.newaxis]
        depths = grasp_group.depths[:, np.newaxis]
        widths = grasp_group.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2))
        # left finger mask
        mask2 = ((targets[:, :, 0] > -self.depth_base) & (targets[:, :, 0] < depths))
        mask3 = (targets[:, :, 1] > -(widths / 2 + self.finger_width))
        mask4 = (targets[:, :, 1] < -widths / 2)
        # right finger mask
        mask5 = (targets[:, :, 1] < (widths / 2 + self.finger_width))
        mask6 = (targets[:, :, 1] > widths / 2)
        # bottom mask
        mask7 = ((targets[:, :, 0] > -(self.depth_base + self.finger_width)) \
                 & (targets[:, :, 0] < -self.depth_base))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)

        collision_mask = np.any((left_mask | right_mask | bottom_mask), axis=-1)

        if not return_empty_grasp:
            return collision_mask

        inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
        empty_mask = inner_mask.sum(axis=-1) < empty_thresh
        return empty_mask | collision_mask