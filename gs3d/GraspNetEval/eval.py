import sys

from PIL import Image

sys.path.append('C:/Users/ARB/PycharmProjects/GraspSkeleton3D')

import os
import gc
import json
import time
import random
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait

import cv2
import torch
import numpy as np
import open3d as o3d
import torch_geometric
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from graspnetAPI.grasp import GraspGroup
from graspnetAPI.graspnet import GraspNet
from graspnetAPI.utils.utils import transform_points
from graspnetAPI.utils.eval_utils import create_table_points

from HGNet.hg_net import HGGQNet
from SPNet.src.networks import V2Net
from SPNet.test_utils import RGBPReader
from gs3d.keypoint_generation import KeypointGeneration
from gs3d.utils.visualization import plot_plane_in_open3d
from gs3d.GraspNetEval.collision_detector import ModelFreeCollisionDetector
from gs3d.GraspNetEval.eval_utils import get_masks, reconstruct_3d_from_mask_and_depth, construct_pcd_from_depth, \
    reconstruct_3d_rgb_from_mask_and_rgb
from gs3d.utils.utils import find_scored_hands, log_time, align_ground_plane, detect_best_plane, rotate_plane_model, \
    plane_transformation

import faulthandler

faulthandler.enable()

ALIGN = False
EVALUATION = True
DEBUG_MODE = True
USE_PARALLEL = False

np.random.seed(42)
random.seed(42)

def return_best_scored_grasps(best_plane, grasp_points, scores, intrinsic_matrix, image_size):
    grasp_net_grasps = []

    grasp_point_score_pairs = list(zip(grasp_points, scores))

    sorted_grasp_point_score_pairs = sorted(grasp_point_score_pairs, key=lambda x: x[1], reverse=True)

    for grasp_point, score in sorted_grasp_point_score_pairs[:200]:
        grasps = find_scored_hands(best_plane, (grasp_point[0], grasp_point[1]),
                                   score, intrinsic_matrix, image_size)
        for g in grasps:
            grasp_net_grasps.append(g)

    return grasp_net_grasps


def process_annotation(pcd, depth, original_depth, rgb_image, camera_pose, masks, best_plane, aligned_best_plane, intrinsic_matrix, step_height,
                       dump_folder, scene_name, camera, ann_id, reward_network, workspace, device_graph,
                       collision_detector_1, collision_detector_2, keypointGenerator, z_transformation):
    try:
        ann_start_time = time.time()
        image_name = '%04d' % ann_id
        grasp_group = GraspGroup()

        for mask_index, mask in enumerate(masks):
            if DEBUG_MODE:
                plt.imshow(mask)
                plt.show()

            masked_grasp_ground = GraspGroup()
            masked_point_cloud = reconstruct_3d_from_mask_and_depth(mask, depth, intrinsic_matrix, workspace)
            masked_original_point_cloud = reconstruct_3d_from_mask_and_depth(mask, original_depth, intrinsic_matrix, workspace, inpainting=False)
            # masked_original_rgb = reconstruct_3d_rgb_from_mask_and_rgb(mask, rgb_image, workspace)

            if ALIGN:
                masked_point_cloud = transform_points(masked_point_cloud, camera_pose)

            masked_pcd = o3d.geometry.PointCloud()
            masked_pcd.points = o3d.utility.Vector3dVector(masked_point_cloud)
            if ALIGN:
                masked_pcd, rotation_matrix = align_ground_plane(masked_pcd, best_plane)

            cl, ind = masked_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
            masked_pcd = masked_pcd.select_by_index(ind)
            masked_pcd = masked_pcd.voxel_down_sample(0.005)

            masked_pcd.transform(z_transformation)
            masked_point_cloud = np.array(masked_pcd.points)
            masked_point_cloud = masked_point_cloud[masked_point_cloud[:, 2] <= 0]

            masked_original_pcd = o3d.geometry.PointCloud()
            masked_original_pcd.points = o3d.utility.Vector3dVector(masked_original_point_cloud)
            # masked_original_pcd.colors = o3d.utility.Vector3dVector(masked_original_rgb)
            if ALIGN:
                masked_original_pcd, rotation_matrix = align_ground_plane(masked_original_pcd, best_plane)

            cl, ind = masked_original_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
            masked_original_pcd = masked_original_pcd.select_by_index(ind)
            masked_original_pcd = masked_original_pcd.voxel_down_sample(0.005)

            masked_original_pcd.transform(z_transformation)
            masked_original_point_cloud = np.array(masked_original_pcd.points)
            masked_original_rgb = np.array(masked_original_pcd.colors)

            if masked_point_cloud.shape[0] < 30:
                continue

            labels = DBSCAN(eps=0.05, min_samples=1).fit_predict(masked_point_cloud)
            main_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
            masked_point_cloud = masked_point_cloud[labels == main_cluster_label]
            masked_pcd.points = o3d.utility.Vector3dVector(masked_point_cloud)

            if masked_point_cloud.shape[0] < 30:
                continue

            keypoint_start_time = time.time()
            selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(masked_point_cloud,
                                                                                           step_height=step_height)
            log_time(keypoint_start_time, scene_id=scene_name, ann_id=ann_id, mask_index=mask_index,
                     message="Keypoint Generation")

            bounding_box = masked_pcd.get_minimal_oriented_bounding_box()
            bounding_box = o3d.geometry.OrientedBoundingBox(bounding_box.center, bounding_box.R,
                                                            bounding_box.extent + np.array([0.05, 0.05, 0.05]))
            cropped_pcd = pcd.crop(bounding_box)

            cropped_pcd.estimate_normals()
            cropped_pcd.orient_normals_towards_camera_location(camera_pose[:3, 3])

            for height_index, height in enumerate(cross_heights):
                keypoint_start_time = time.time()
                final_grasp_pairs, scores = keypointGenerator.generate_scored_grasp_pairs(
                    height,
                    selected_points[height_index],
                    cropped_pcd)
                log_time(keypoint_start_time, scene_id=scene_name, ann_id=ann_id, mask_index=mask_index,
                         height_index=height_index, message="Scored Grasp Pose Generation")

                if len(final_grasp_pairs) > 0:
                    final_grasp_pairs = np.array(final_grasp_pairs)

                    grasp_pose_start_time = time.time()
                    g_group = return_best_scored_grasps(aligned_best_plane, final_grasp_pairs, scores, intrinsic_matrix,
                                                        depth.shape)
                    for g in g_group:
                        masked_grasp_ground.add(g)
                    log_time(grasp_pose_start_time, scene_id=scene_name, ann_id=ann_id, mask_index=mask_index,
                             height_index=height_index, message="Grasp Pose Tune")

            grasp_filter_start_time = time.time()
            masked_grasp_ground = masked_grasp_ground.nms(0.03, 30.0 / 180 * np.pi)
            collision_mask = collision_detector_1.detect(masked_grasp_ground, return_empty_grasp=False)
            masked_grasp_ground.grasp_group_array = np.array(masked_grasp_ground.grasp_group_array)
            # masked_grasp_ground.grasp_group_array[:, 0] = 0
            collision_indices = np.where(collision_mask)[0]
            masked_grasp_ground.grasp_group_array[collision_indices, 0] -= 10
            # masked_grasp_ground = masked_grasp_ground[~collision_mask]
            collision_mask = collision_detector_2.detect(masked_grasp_ground, empty_thresh=10, return_empty_grasp=True)
            collision_indices = np.where(collision_mask)[0]
            masked_grasp_ground.grasp_group_array[collision_indices, 0] -= 10
            # masked_grasp_ground = masked_grasp_ground[~collision_mask]
            log_time(grasp_filter_start_time, scene_id=scene_name, ann_id=ann_id, mask_index=mask_index,
                     message="Grasp Filtering and Collision Detection")

            reward_start_time = time.time()

            apply_reward_to_grasps(masked_grasp_ground, masked_original_point_cloud, masked_original_rgb, reward_network, grasp_group, device_graph, z_transformation)

            log_time(reward_start_time, scene_id=scene_name, ann_id=ann_id, message="Apply reward to grasps")

        grasp_group.transform(np.linalg.inv(z_transformation))
        save_grasp_group(grasp_group, camera_pose, dump_folder, scene_name, camera, image_name)
        log_time(ann_start_time, scene_id=scene_name, ann_id=ann_id, message="✔   Processing annotation")

    except Exception as e:
        print(f"Exception Occurred in annotation {ann_id}: {e}")
        import traceback
        traceback.print_exc()


def save_grasp_group(grasp_group, camera_pose, dump_folder, scene_name, camera, image_name):
    if ALIGN:
        grasp_group.transform(np.linalg.inv(camera_pose))

    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    if not os.path.exists(f"{dump_folder}/{scene_name}/"):
        os.makedirs(f"{dump_folder}/{scene_name}/")
    if not os.path.exists(f"{dump_folder}/{scene_name}/{camera}"):
        os.makedirs(f"{dump_folder}/{scene_name}/{camera}")

    grasp_group.save_npy(f"{dump_folder}/{scene_name}/{camera}/{image_name}.npy")


def apply_reward_to_grasps(masked_grasp_ground, masked_point_cloud, masked_original_rgb, reward_network, grasp_group, device, z_transformation):
    if len(masked_grasp_ground) == 0:
        return

    # inv_z_transformation = np.linalg.inv(z_transformation)
    # masked_grasp_ground.transform(inv_z_transformation)
    # masked_point_cloud = (masked_point_cloud @ inv_z_transformation[:3, :3].T) + inv_z_transformation[:3, 3]

    # centers = masked_grasp_ground.translations
    # rotation_matrices = masked_grasp_ground.rotation_matrices
    #
    # depths = masked_grasp_ground.depths
    # widths = masked_grasp_ground.widths
    #
    # grasp_configs = np.hstack([centers, rotation_matrices.reshape(-1, 9), depths[:, None], widths[:, None]])
    #
    # # Convert point cloud to graph representation
    # pc = torch.tensor(masked_point_cloud, dtype=torch.float32).cuda()
    # rgb = torch.tensor(masked_original_rgb, dtype=torch.float32).cuda()
    # graph = torch_geometric.data.batch.Batch.from_data_list([torch_geometric.data.Data(xyz=pc, rgb=rgb)], follow_batch=['xyz'])
    #
    # # Convert grasp configurations to tensor
    # grasp_configs_tensor = torch.tensor(grasp_configs, dtype=torch.float32).cuda()
    #
    # # Create graph indices tensor
    # graph_indices = torch.zeros(grasp_configs_tensor.size(0), dtype=torch.int64).cuda()
    #
    # # Predict scores using the reward network
    # with torch.no_grad():
    #     predicted_scores, center_scores, _ = reward_network(
    #         grasp_configs_tensor, graph, graph_indices
    #     )
    #
    # # Assign predicted scores to the masked grasp group
    # predicted_scores_array = predicted_scores.cpu().numpy()
    # masked_grasp_ground.grasp_group_array[:, 0] += predicted_scores_array * 10

    masked_grasp_ground.sort_by_score()
    for masked_grasp in masked_grasp_ground[:10]:
        grasp_group.add(masked_grasp)

    # grasp_group.transform(z_transformation)


def load_networks(config, device_graph, device_depth_completion):
    # checkpoint_path = config['checkpoint_path']
    # reward_network = RewardNetwork(node_feature_size=3, action_size=7, descriptor_size=64, fc_size=64).to(device_graph)
    # checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # reward_network.load_state_dict(checkpoint['model_state_dict'])
    # reward_network.eval()

    reward_network = HGGQNet().cuda()
    checkpoint = torch.load('HGNet/checkpoints/checkpoint_epoch10.tar', map_location='cpu', weights_only=False)
    reward_network.load_state_dict(checkpoint['model_state_dict'])
    reward_network.eval()

    depth_network = V2Net([192, 384, 768, 1536], [3, 3, 27, 3], 0.2, 'CNX')
    depth_checkpoint = torch.load('SPNet/checkpoints/Large.pth', map_location='cpu', weights_only=False)
    depth_network.load_state_dict(depth_checkpoint['network'])
    depth_network.eval()

    # return reward_network.to(device_graph), depth_network.to(device_depth_completion)
    return reward_network, depth_network.to(device_depth_completion)


def denoised_depth_image(graspnet_root, scene_name, camera, image_name, intrinsic_matrix, z_transformation):
    original_depth_image = cv2.imread(
        str(os.path.join(graspnet_root, 'scenes', scene_name, camera, 'depth', f"{image_name}.png")),
        cv2.IMREAD_UNCHANGED)

    depth_image = original_depth_image.copy()

    height, width = depth_image.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid_mask = depth_image > 0
    z = depth_image[valid_mask] / 1000.0
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    points = np.vstack((x, y, z)).T
    point_cloud_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_point_cloud_homogeneous = point_cloud_homogeneous @ z_transformation.T
    points = transformed_point_cloud_homogeneous[:, :3]
    under_table = points[:, 2] > 0.0
    X = np.argwhere(valid_mask)
    depth_image[X[under_table, 0], X[under_table, 1]] = 0

    scaling_factor = 0.1
    zoom_factors = (scaling_factor, scaling_factor)
    small_depth_image = zoom(depth_image, zoom_factors, order=0)

    new_intrinsic_matrix = intrinsic_matrix.copy()
    new_intrinsic_matrix[0, 0] *= scaling_factor
    new_intrinsic_matrix[0, 2] *= scaling_factor
    new_intrinsic_matrix[1, 1] *= scaling_factor
    new_intrinsic_matrix[1, 2] *= scaling_factor
    height, width = small_depth_image.shape
    fx, fy = new_intrinsic_matrix[0, 0], new_intrinsic_matrix[1, 1]
    cx, cy = new_intrinsic_matrix[0, 2], new_intrinsic_matrix[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid_mask = small_depth_image > 0
    X = np.argwhere(valid_mask)

    z = small_depth_image[valid_mask] / 1000.0
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    points = np.vstack((x, y, z)).T

    db = DBSCAN(eps=0.02, min_samples=1).fit(points)
    small_mask = np.ones_like(small_depth_image, dtype=np.uint8)
    main_cluster_label = np.argmax(np.bincount(db.labels_[db.labels_ >= 0]))
    main_cluster_labels = db.labels_ == main_cluster_label
    small_mask[X[main_cluster_labels, 0], X[main_cluster_labels, 1]] = 0

    zoom_factors = (depth_image.shape[0] / small_mask.shape[0],
                    depth_image.shape[1] / small_mask.shape[1])
    mask_full_size = zoom(small_mask, zoom_factors, order=0)

    depth_image[mask_full_size == 1] = 0
    return depth_image, original_depth_image


def process_scene(scene_number, config_path, step_height, camera, device_graph, device_depth_completion):
    try:
        scene_start_time = time.time()
        print(f"Processing scene {scene_number} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        if scene_number % 4 == 0:
            device_depth_completion = 'cuda' if torch.cuda.is_available() else 'cpu'

        reward_network, depth_network = load_networks(config, device_graph, device_depth_completion)
        dump_folder = config['dump_folder']
        graspnet_root = config['graspnet_root']

        grasp_net = GraspNet(graspnet_root, camera=camera, split='custom')
        keypointGenerator = KeypointGeneration(DEBUG_MODE, True)

        for ann_id in range(256):
            annotation_start_time = time.time()
            image_name = f"{ann_id:04d}"
            scene_name = f"scene_{scene_number:04d}"
            if os.path.exists(f"{dump_folder}/{scene_name}/{camera}/{image_name}.npy"):
                continue

            table = create_table_points(0.02, 0.02, 0.01)
            intrinsic_matrix = np.load(os.path.join(graspnet_root, 'scenes', scene_name, camera, 'camK.npy'))
            camera_pose = np.load(os.path.join(graspnet_root, 'scenes', scene_name, camera, 'camera_poses.npy'))[ann_id]
            align_mat = np.load(os.path.join(graspnet_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
            masks = get_masks(os.path.join(graspnet_root, 'scenes', scene_name, camera, 'label', f"{image_name}.png"))

            if ALIGN:
                camera_pose = np.matmul(align_mat, camera_pose)

            table = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
            table_plane = detect_best_plane(table, 4)

            z_transformation = plane_transformation(table_plane, (0, 0, 1.0, 0.0))

            original_plane = (0, 0, 1, 0)

            log_time(annotation_start_time, scene_id=scene_number, ann_id=ann_id, message="Plane Detection")

            reconstruction_start_time = time.time()
            # original_depth = imread(os.path.join(graspnet_root, 'scenes', scene_name, camera, 'depth', f"{image_name}.png"))
            # original_size = original_depth.shape
            original_size = (720, 1280)

            workspace = grasp_net.loadWorkSpace(scene_number, camera, ann_id)
            # workspace = None

            rgb_image = np.array(Image.open(str(os.path.join(graspnet_root, 'scenes', scene_name, camera, 'rgb', f"{image_name}.png"))), dtype=np.float32) / 255.0

            depth, original_depth = denoised_depth_image(graspnet_root, scene_name, camera, image_name, intrinsic_matrix,
                                         z_transformation)
            rgbd_reader = RGBPReader()
            rgb, raw, hole_raw = rgbd_reader.read_data(
                None,
                os.path.join(graspnet_root, 'scenes', scene_name, camera, 'rgb', f"{image_name}.png"),
                depth
            )
            raw = raw.to(device_depth_completion)
            rgb = rgb.to(device_depth_completion)
            hole_raw = hole_raw.to(device_depth_completion)
            pred = depth_network(rgb, raw, hole_raw)
            raw[hole_raw == 0] = pred[hole_raw == 0]

            # if workspace is not None:
            #     (x1, y1, x2, y2) = workspace
            #     original_size = (int(y2 - y1), int(x2 - x1))

            depth = torch.nn.functional.interpolate(raw, size=original_size,
                                                    mode='nearest-exact').squeeze().cpu().detach().numpy()
            depth = np.clip(depth * 65535., 0, 65535).astype(np.int32)
            colors = grasp_net.loadRGB(sceneId=scene_number, camera=camera, annId=ann_id).astype(
                np.float32) / 255.0

            # if workspace is not None:
            #     (x1, y1, x2, y2) = workspace
            #     colors = colors[y1:y2, x1:x2, :]
            pcd = construct_pcd_from_depth(original_size, intrinsic_matrix, depth, colors, None)


            log_time(reconstruction_start_time, scene_id=scene_number, ann_id=ann_id,
                     message="Point Cloud Reconstruction")

            pcd.transform(z_transformation)
            pcd = pcd.voxel_down_sample(0.005)
            pcd_points = np.asarray(pcd.points)
            pcd.points = o3d.utility.Vector3dVector(pcd_points[pcd_points[:, 2] <= 0])

            if DEBUG_MODE:
                plot_plane_in_open3d(pcd, original_plane)

            aligned_best_plane = original_plane
            if ALIGN:
                pcd, rotation_matrix = align_ground_plane(pcd, original_plane)
                aligned_best_plane = rotate_plane_model(original_plane, rotation_matrix)

            original_pcd = grasp_net.loadScenePointCloud(scene_number, camera, ann_id)
            original_pcd = original_pcd.voxel_down_sample(0.005)
            original_pcd.transform(z_transformation)
            original_pcd_points = np.asarray(original_pcd.points)
            original_pcd.points = o3d.utility.Vector3dVector(original_pcd_points[original_pcd_points[:, 2] <= 0])
            collision_detector_1 = ModelFreeCollisionDetector(original_pcd.points)
            collision_detector_2 = ModelFreeCollisionDetector(pcd.points)
            process_annotation(
                pcd, depth, original_depth, rgb_image, camera_pose, masks, original_plane, aligned_best_plane,
                intrinsic_matrix, step_height, dump_folder, scene_name, camera, ann_id,
                reward_network, workspace, device_graph, collision_detector_1, collision_detector_2,
                keypointGenerator, z_transformation
            )

            # Clean up to free memory
            del rgb, raw, hole_raw
            del depth, original_depth, masks, pcd
            torch.cuda.empty_cache()
            gc.collect()

            if not USE_PARALLEL and EVALUATION:
                eval_start_time = time.time()
                from graspnetAPI.graspnet_eval import GraspNetEval
                grasp_net_eval = GraspNetEval(root=graspnet_root, camera=camera,
                                              split='custom') if not USE_PARALLEL and EVALUATION else None
                acc = grasp_net_eval.eval_annotations(scene_id=scene_number,
                                                      ann_ids=[ann_id],
                                                      dump_folder=dump_folder,
                                                      return_list=False,
                                                      vis=DEBUG_MODE or False)
                log_time(eval_start_time, scene_id=scene_number, ann_id=ann_id, message="GraspNet Evaluation")
                print(f'Mean Accuracy for Scene {scene_number}: {np.mean(acc)}')

        log_time(scene_start_time, scene_id=scene_number, message="\u2714   Processing scene ")
    except Exception as e:
        print("Exception Occurred:")
        print(e)
        traceback.print_exc()

    finally:
        torch.cuda.empty_cache()
        gc.collect()
        del reward_network, depth_network


if __name__ == "__main__":
    STEP_HEIGHT = 0.9
    device_depth_completion = 'cpu'
    device_graph = 'cuda' if torch.cuda.is_available() else 'cpu'
    overall_start_time = time.time()

    print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config_path = 'pathes.json'
    cameras = ['kinect']

    for camera in cameras:
        if USE_PARALLEL:
            with ProcessPoolExecutor(max_workers=12) as executor:
                futures = [
                    executor.submit(process_scene, scene_number, config_path, STEP_HEIGHT, camera, device_graph,
                                    device_depth_completion)
                    for scene_number in range(100, 190)
                ]
                wait(futures)
        else:
            for scene_number in range(146, 190):
                process_scene(scene_number, config_path, STEP_HEIGHT, camera, device_graph, device_depth_completion)

    log_time(overall_start_time, message="Total script execution")
