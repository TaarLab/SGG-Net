import os
import csv
import math
import asyncio
import open3d as o3d
from datetime import datetime
import o3d
import zarr
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from gs3d.RL.model import RewardNetwork
from gs3d.utils.lru_cache import LRUCache
from gs3d.keypoint_generation import KeypointGeneration
from gs3d.utils.model_utils import pc_to_graph_from_zarr
from gs3d.utils.utils import cleanup_shapes, get_obj_files, add_obj_file_to_sim, setGripperData, \
    moveToPose_viaIK, moveToConfig_viaFK, wait_until_object_stops, \
    get_point_cloud, initialize_simulation, find_best_gripper_hand, vector_to_quaternion, check_collision

device = 'cuda' if torch.cuda.is_available() else 'cpu'


async def main():
    client = RemoteAPIClient()
    sim, simIK = initialize_simulation(client)
    # obj_directory = "E:/TaarLab/3D-Skeleton/egad_eval_set/"
    obj_directory = r"E:\TaarLab\meshses_folder\meshes\meshes\mymeshes"
    obj_files = get_obj_files(obj_directory)

    # Initialize robot joint values
    simJoints = [sim.getObject('/UR5/joint', {'index': i}) for i in range(6)]
    simTip = sim.getObject('/UR5/ikTip')
    simTarget = sim.getObject('/UR5/ikTarget')
    modelBase = sim.getObject('/UR5')
    gripperHandle = sim.getObject('/UR5/openCloseJoint')

    ikEnv = simIK.createEnvironment()

    # Prepare the IK group
    ikGroup = simIK.createGroup(ikEnv)
    simIK.addElementFromScene(ikEnv, ikGroup, modelBase, simTip, simTarget, simIK.constraint_pose)

    # Define FK movement data
    vel, accel, jerk = 180, 40, 80
    maxVel, maxAccel, maxJerk = [vel * math.pi / 180 for _ in range(6)], [accel * math.pi / 180 for _ in range(6)], [
        jerk * math.pi / 180 for _ in range(6)]

    # Define IK movement data
    ikMaxVel, ikMaxAccel, ikMaxJerk = [0.4, 0.4, 0.4, 1.8], [0.8, 0.8, 0.8, 0.9], [0.6, 0.6, 0.6, 0.8]

    placeConfig = [1.29158355133319, 0.3106908723883177, 0.0, 3.5940672100997455, 1.5705821039338792, 4.433069596719524]

    data = {
        'ikEnv': ikEnv,
        'ikGroup': ikGroup,
        'tip': simTip,
        'target': simTarget,
        'joints': simJoints
    }

    keypointGenerator = KeypointGeneration(False, True)

    checkpoint_path = 'checkpoints_64_cos/reward_network_epoch_227_loss_0.1731.pth'

    node_feature_size = 3
    action_size = 7
    reward_network = RewardNetwork(node_feature_size=node_feature_size, action_size=action_size, descriptor_size=64,
                                   fc_size=64).to(device)

    zarr_store = zarr.open('dexnet_dataset_eval.zarr', mode='a')

    report_data = []

    for obj_file in obj_files:
        fid = obj_file

        sim.stopSimulation()

        shape_handles = []
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

        if skip_file:
            continue

        labels = DBSCAN(eps=0.05, min_samples=1).fit_predict(point_cloud)
        main_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
        point_cloud = point_cloud[labels == main_cluster_label]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])

        pc_key = f'point_cloud_{fid}'
        # if pc_key not in zarr_store:
        #     zarr_store.create_dataset(pc_key, data=point_cloud, compressor=zarr.Blosc())

        selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(good_points)
        grasp_points = []

        indices = [len(selected_points) // 2, 1, -2]
        if len(selected_points) < len(indices):
            indices = [] if len(selected_points) > 0 else [0]
        for height_index in indices:
            height = cross_heights[height_index]
            # finalGraspPoses, skeleton_points = keypointGenerator.generate_grasp_poses(height_index, point_cloud)
            # if finalGraspPoses == None:
            #     continue
            # grasp_points += [((x1, y1, height), (x2, y2, height)) for ((x1, y1), (x2, y2)) in finalGraspPoses]

            finalGraspPoses, scores, skeleton_points = keypointGenerator.generate_scored_grasp_pairs(
                height_index,
                pcd,
                point_cloud)

            grasp_points = np.array(finalGraspPoses)


        grasp_poses = []
        for i, grasp_point in enumerate(grasp_points):
            angle, point = check_collision(grasp_point[0], grasp_point[1], np.array([0, 0, 0.4]))
            if angle is not None and not np.all(np.array(grasp_point[0]) - np.array(grasp_point[1]) == 0.0):
                grasp_pose = np.concatenate(
                    [point, vector_to_quaternion(angle, np.array(grasp_point[0]) - np.array(grasp_point[1]))])
                # grasps = find_best_gripper_hand(pcd, pcd, None, None, grasp_point, False)
                # for grasp in grasps:
                #     target_rotation = Rotation.from_matrix(grasp.rotation_matrix.T)
                #     grasp_pose = np.concatenate([grasp.translation, target_rotation.as_quat()])

                action_tensor = torch.tensor(grasp_pose).float().unsqueeze(0).to(device)

                node_features, edge_index, batch_index = pc_to_graph_from_zarr(
                    zarr_store, cache, [pc_key]
                )

                node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
                edge_index = torch.tensor(edge_index).to(device)
                batch_index = torch.tensor(batch_index, dtype=torch.int64).to(device)

                predicted_reward = reward_network(node_features, edge_index, batch_index, action_tensor)
                predicted_reward = predicted_reward.item()

                print(f"[{i}] Prediction: {predicted_reward:.4f}")

                if predicted_reward > best_reward:
                    best_reward = predicted_reward
                    grasp_poses = [grasp_pose]
                    grasp_points_f = [np.concatenate(grasp_point)]
                # if predicted_reward > 1.0:
                #     grasp_poses.append(grasp_pose)
                #     grasp_points_f.append(np.concatenate(grasp_point))

        successful_grasps = 0
        total_grasps = len(grasp_poses)
        trial_data = []

        for grasp_pose in grasp_poses:
            client = RemoteAPIClient()

            sim = client.require('sim')
            simIK = client.require('simIK')
            sim.startSimulation()

            wait_until_object_stops(sim, shape_handles[0])
            sim.setEngineFloatParam(sim.newton_body_kineticfriction,shape_handles[0],1.2)
            sim.setEngineFloatParam(sim.newton_body_staticfriction,shape_handles[0],1.2)
            starting_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

            setGripperData(sim, gripperHandle, True)

            moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)

            setGripperData(sim, gripperHandle, False)

            # NO MAGIC
            moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)

            await asyncio.sleep(0.5)

            moveToConfig_viaFK(sim, maxVel, maxAccel, maxJerk, placeConfig, data)

            ending_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

            is_successful = ending_height - starting_height > 0.2

            if is_successful:
                successful_grasps += 1

            trial_data.append((grasp_pose, is_successful, grasp_point))

            sim.stopSimulation()

            print(f"[+] Is Grasp Successful: {is_successful}")

        success_rate = successful_grasps / total_grasps if total_grasps > 0 else 0.0
        print(f"Object: {obj_file}, Success Rate: {success_rate:.2f}")

        report_data.append({
            'Object File': obj_file,
            'Total Grasps': total_grasps,
            'Successful Grasps': successful_grasps,
            'Success Rate': success_rate
        })

        cleanup_shapes(sim, shape_handles)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')

    results_dir = f'results/{current_datetime}'

    os.makedirs(results_dir, exist_ok=True)

    save_report_to_csv(report_data, os.path.join(results_dir, 'object_success_rates.csv'))
    print_report_to_console(report_data)

    overall_success_rate = calculate_overall_success_rate(report_data)
    save_overall_success_rate_to_csv(overall_success_rate, os.path.join(results_dir, 'overall_success_rate.csv'))


def save_report_to_csv(data, filename):
    keys = data[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


def print_report_to_console(data):
    df = pd.DataFrame(data)
    print("\nGrasp Evaluation Report:")
    print(df.to_string(index=False))


def calculate_overall_success_rate(data):
    total_attempts = sum([item['Total Grasps'] for item in data])
    total_successes = sum([item['Successful Grasps'] for item in data])
    overall_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0
    print(f"\nOverall Success Rate: {overall_success_rate:.2f}")
    return overall_success_rate


def save_overall_success_rate_to_csv(overall_success_rate, filename):
    with open(filename, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Overall Success Rate'])
        writer.writerow([overall_success_rate])


if __name__ == "__main__":
    asyncio.run(main())
