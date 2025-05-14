import time

import zarr
import asyncio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from sklearn.cluster import DBSCAN

from gs3d.keypoint_generation import KeypointGeneration
from gs3d.utils.utils import cleanup_shapes, get_obj_files, add_obj_file_to_sim, setGripperData, \
    moveToPose_viaIK, moveToConfig_viaFK, wait_until_object_stops, \
    setup_copeliasim, get_point_cloud, find_best_gripper_hand


async def main():
    sim, gripperHandle, forward_config, inverse_config, placeConfig, data = setup_copeliasim()
    # obj_directory = "E:/TaarLab/3D-Skeleton/egad_train_set"
    # obj_directory = "E:/TaarLab/3D-Skeleton"
    obj_directory = r"C:\Users\ARB\Desktop\EGAD\egad_train_set"
    obj_files = get_obj_files(obj_directory, '_0')

    keypointGenerator = KeypointGeneration(False, True)

    zarr_store = zarr.open('dataset.zarr', mode='a')

    shape_handles = []
    for obj_file in obj_files:
        fid = obj_file

        pc_key = f'point_cloud_{fid}'
        if pc_key in zarr_store.keys():
            cleanup_shapes(sim, shape_handles)
            continue

        sim.stopSimulation()

        skip_file = False
        while True:
            try:
                shape_handles = add_obj_file_to_sim(fid, obj_file, obj_directory, sim, 1)
                if shape_handles is None:
                    skip_file = True
                    break

                point_cloud = get_point_cloud(sim)
                break
            except Exception as e:
                print(e)
                cleanup_shapes(sim, shape_handles)

        if skip_file or point_cloud.shape[0] == 0:
            cleanup_shapes(sim, shape_handles)
            continue

        labels = DBSCAN(eps=0.005, min_samples=10).fit_predict(point_cloud)
        main_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
        point_cloud = point_cloud[labels == main_cluster_label]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd = pcd.voxel_down_sample(0.01)

        point_cloud = np.array(pcd.points)

        zarr_store.create_dataset(pc_key, data=point_cloud, compressor=zarr.Blosc())

        selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(point_cloud)
        grasp_points = []

        indices = [len(selected_points) // 2, 1, -2]
        if len(selected_points) < len(indices):
            indices = [] if len(selected_points) > 0 else [0]
        skeleton_points_3d = []
        for height_index in indices:
            height = cross_heights[height_index]

            keypointGenerator = KeypointGeneration(False, True)
            finalGraspPoses, skeleton_points = keypointGenerator.generate_grasp_poses(height_index, point_cloud)
            skeleton_points_3d += skeleton_points

            if finalGraspPoses == None:
                continue

            grasp_points += [((x1, y1, height), (x2, y2, height)) for ((x1, y1), (x2, y2)) in finalGraspPoses]

        grasp_poses = []
        grasp_points_f = []
        for i, grasp_point in enumerate(grasp_points):
            grasps = find_best_gripper_hand(pcd, pcd, None, None, grasp_point, False)
            for grasp in grasps:
                target_rotation = Rotation.from_matrix(grasp.rotation_matrix.T)
                grasp_poses.append(np.concatenate([grasp.translation, target_rotation.as_quat()]))
                grasp_points_f.append(np.concatenate(grasp_point))

        successful_grasps = 0
        total_grasps = len(grasp_poses)
        trial_data = []

        for i, (grasp_pose, grasp_point) in enumerate(zip(grasp_poses, grasp_points_f)):
            client = RemoteAPIClient()

            sim = client.require('sim')
            simIK = client.require('simIK')
            sim.startSimulation()

            wait_until_object_stops(sim, shape_handles[0])

            starting_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

            setGripperData(sim, gripperHandle, True)

            moveToPose_viaIK(sim, simIK, inverse_config[0], inverse_config[1], inverse_config[2], list(grasp_pose),
                             data)

            setGripperData(sim, gripperHandle, True, 0, 0)

            time.sleep(0.2)

            setGripperData(sim, gripperHandle, False)

            time.sleep(0.2)

            moveToConfig_viaFK(sim, forward_config[0], forward_config[1], forward_config[2], placeConfig, data)

            time.sleep(1.5)

            ending_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

            is_successful = ending_height - starting_height > 0.2

            if is_successful:
                successful_grasps += 1

            trial_data.append((grasp_pose, is_successful, grasp_point))

            sim.stopSimulation()

            print(f"[{i}] Is Grasp Successful: " + str(is_successful))

        if len(grasp_points_f) > 0:
            dtype = np.dtype([
                ('grasp_pose', np.float32, grasp_poses[0].shape),
                ('is_successful', np.bool_),
                ('grasp_points', np.float32, grasp_points_f[0].shape)
            ])

            trial_data_array = np.array(trial_data, dtype=dtype)

            trial_data_key = f'trial_data_{pc_key}'
            zarr_store.create_dataset(trial_data_key, data=trial_data_array, compressor=zarr.Blosc())

        success_rate = successful_grasps / total_grasps if total_grasps > 0 else 0.0
        print(f"Object: {obj_file}, Success Rate: {success_rate}")

        cleanup_shapes(sim, shape_handles)

if __name__ == "__main__":
    asyncio.run(main())
