import os
import csv
import time
import math
import random
import socket
import asyncio
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import pandas as pd
import open3d as o3d
import torch_geometric
from scipy.spatial.transform import Rotation
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from HGNet.hg_net import HGGQNet
from graspnetAPI.grasp import GraspGroup
from gs3d.keypoint_generation import KeypointGeneration
from gs3d.GraspNetEval.collision_detector import ModelFreeCollisionDetector
from gs3d.utils.utils import cleanup_shapes, get_obj_files, add_obj_file_to_sim, \
    get_point_cloud, initialize_simulation, find_scored_hands, \
    wait_until_object_stops, setGripperData, moveToPose_viaIK, moveToConfig_viaFK

STEP_HEIGHT = 0.9
NUM_WORKER = 10

import faulthandler
faulthandler.enable()

def apply_model(hggNet, grasp_ground, point_cloud):
    centers = grasp_ground.translations
    rotation_matrices = grasp_ground.rotation_matrices

    depths = grasp_ground.depths
    widths = grasp_ground.widths

    grasp_configs = np.hstack([centers, rotation_matrices.reshape(-1, 9), depths[:, None], widths[:, None]])

    # Convert point cloud to graph representation
    pc = torch.tensor(point_cloud[:, :3], dtype=torch.float32).cuda()
    rgb = torch.tensor(np.empty((3, 1)), dtype=torch.float32).cuda()
    graph = torch_geometric.data.batch.Batch.from_data_list([torch_geometric.data.Data(xyz=pc, rgb=rgb)], follow_batch=['xyz'])

    # Convert grasp configurations to tensor
    grasp_configs_tensor = torch.tensor(grasp_configs, dtype=torch.float32).cuda()

    # Create graph indices tensor
    graph_indices = torch.zeros(grasp_configs_tensor.size(0), dtype=torch.int64).cuda()

    # Predict scores using the reward network
    with torch.no_grad():
        predicted_scores, center_scores, _ = hggNet(
            grasp_configs_tensor, graph, graph_indices
        )

    # Assign predicted scores to the masked grasp group
    predicted_scores_array = predicted_scores.cpu().numpy()
    grasp_ground.grasp_group_array[:, 0] += predicted_scores_array * 10

def save_report_to_csv(data, filename):
    try:
        df = pd.DataFrame(data)

        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)

        print(f"Report successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the report: {e}")


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


async def start_coppeliasim(scene_file=None, headless=False):
    """Start CoppeliaSim in headless mode with ZMQ remote API."""

    while True:
        port = random.randint(23005, 24000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('127.0.0.1', port))
            if result != 0:
                break

    print(f"Starting CoppeliaSim on port {port}")

    command = ["C:\\Program Files\\CoppeliaRobotics\\CoppeliaSimEdu\\coppeliaSim.exe", f'-GzmqRemoteApi.rpcPort={port}']
    if headless:
        command += ['-H']
    if scene_file:
        command += ['-f', scene_file]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    while True:
        try:
            _ = await asyncio.wait_for(process.stdout.readline(), timeout=2)
            await asyncio.sleep(0.01)
        except:
            break
    print(f"CoppeliaSim on port {port} Started")

    return process, port

async def evaluate_grasps_for_object(obj_file, hggNet, keypointGenerator):
    process, port = await start_coppeliasim(r'C:\Users\ARB\PycharmProjects\GraspSkeleton3D\grasp_skeleton_env.ttt')
    try:
        client = RemoteAPIClient(port=port)
        sim, simIK = initialize_simulation(client)

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
        vel, accel, jerk = 10, 40, 80
        maxVel, maxAccel, maxJerk = [vel * math.pi / 180 for _ in range(6)], [accel * math.pi / 180 for _ in range(6)], [
            jerk * math.pi / 180 for _ in range(6)]

        # Define IK movement data
        ikMaxVel, ikMaxAccel, ikMaxJerk = [0.4, 0.4, 0.4, 0.2], [0.8, 0.8, 0.8, 0.9], [0.6, 0.6, 0.6, 0.8]

        placeConfig = [0., 0., 0.0, 0., 0., 0.]

        data = {
            'ikEnv': ikEnv,
            'ikGroup': ikGroup,
            'tip': simTip,
            'target': simTarget,
            'joints': simJoints
        }

        report_data = []
        shape_handles = []
        skip_file = False
        while True:
            try:
                shape_handles = add_obj_file_to_sim(obj_file, client, sim)

                if shape_handles is None:
                    skip_file = True
                    break

                point_cloud = get_point_cloud(sim, port)
                break
            except Exception as e:
                print(e)
                cleanup_shapes(sim, shape_handles)

        if skip_file:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])
        pcd = pcd.voxel_down_sample(0.005)

        selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(point_cloud, STEP_HEIGHT)
        grasp_points = []
        scores = []

        for height_index, height in enumerate(cross_heights):
            final_grasp_pairs, scores = keypointGenerator.generate_scored_grasp_pairs(height, selected_points[height_index],
                                                                                      pcd)

            if len(final_grasp_pairs) > 0:
                grasp_points = np.array(final_grasp_pairs)

        grasp_group = GraspGroup()
        for i, (grasp_point, score) in enumerate(zip(grasp_points, scores)):
            grasps = find_scored_hands((0, 0, 1, 0), grasp_point, score,
                                       None, None, (0, 0, 10),
                                       True, [0.01, 0.02], [0.05], True)
            for grasp in grasps:
                grasp_group.add(grasp)

        collision_detector = ModelFreeCollisionDetector(pcd.points)
        collision_mask = collision_detector.detect(grasp_group, empty_thresh=10, return_empty_grasp=True)
        # grasp_group = grasp_group[~collision_mask]
        collision_indices = np.where(collision_mask)[0]
        grasp_group.grasp_group_array[collision_indices, 0] -= 10

        apply_model(hggNet, grasp_group, point_cloud)
        grasp_group.sort_by_score()

        # o3d.visualization.draw_geometries([pcd] + grasp_group[:10].to_open3d_geometry_list())

        # Collect grasp poses
        grasp_poses = []
        for grasp in grasp_group[:3]:
            g_rotation_matrix = grasp.rotation_matrix
            if np.linalg.det(g_rotation_matrix) < 0:
                g_rotation_matrix[:, 2] *= -1
            translation = grasp.translation
            translation[2] -= grasp.depth
            translation[2] -= 0.01
            translation[2] = max(translation[2], 0.04)
            grasp_poses.append(np.concatenate([translation, Rotation.from_matrix(g_rotation_matrix).as_quat()]))

        successful_grasps = 0
        unreachable_grasps = 0
        unstable_shape = 0
        total_grasps = 0

        for grasp_pose in grasp_poses:
            total_grasps += 1
            client = RemoteAPIClient(port=port)
            sim = client.require('sim')
            simIK = client.require('simIK')

            def attempt_grasp():
                sim.startSimulation()

                wait_until_object_stops(sim, shape_handles[0])

                starting_pose = sim.getObjectPose(shape_handles[0], sim.handle_world)

                starting_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

                setGripperData(sim, gripperHandle, True)

                moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)

                threshold = 0.01
                start_time = time.time()
                while True:
                    simTip_pose = sim.getObjectPose(simTip, sim.handle_world)
                    pose_difference = [abs(simTip_pose[i] - grasp_pose[i]) for i in range(len(grasp_pose))]
                    is_within_threshold = all(diff <= threshold for diff in pose_difference)

                    if is_within_threshold:
                        break

                    if time.time() - start_time >= 4:
                        print("[!] Configuration is not reachable within 3 seconds.")
                        moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)
                        break

                pose_before_grasp = sim.getObjectPose(shape_handles[0], sim.handle_world)
                pose_difference = [abs(starting_pose[i] - pose_before_grasp[i]) for i in range(len(pose_before_grasp))]
                is_within_threshold = all(diff <= threshold for diff in pose_difference)
                if not is_within_threshold:
                    print("[!] Configuration is still not reachable after retry.")
                    sim.stopSimulation()
                    return False, False, True

                setGripperData(sim, gripperHandle, False)

                time.sleep(3)

                moveToConfig_viaFK(sim, maxVel, maxAccel, maxJerk, placeConfig, data)

                ending_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

                sim.stopSimulation()

                return ending_height - starting_height > 0.2, False, False

            is_successful = False

            for attempt in range(3):
                is_successful, is_unreachable, is_unstable_shape = attempt_grasp()
                if is_successful:
                    successful_grasps += 1
                    break
                if is_unreachable:
                    unreachable_grasps += 1
                    break
                if is_unstable_shape:
                    unstable_shape += 1
                    break
                time.sleep(1)

            print(f"[+] {obj_file} Is Grasp Successful: {is_successful}")

            if successful_grasps > 0 or unstable_shape > 0:
                break

        success_rate = successful_grasps / total_grasps if total_grasps > 0 else 0.0
        print(f"Object: {obj_file}, Success Rate: {success_rate:.2f}")

        report_data.append({
            'Object File': obj_file,
            'Total Grasps': total_grasps,
            'Unreachable Grasps': unreachable_grasps,
            'Unstable Shapes': unstable_shape,
            'Successful Grasps': successful_grasps,
        })

        cleanup_shapes(sim, shape_handles)

        return report_data
    except:
        traceback.print_exc()
    finally:
        process.terminate()


def execute_evaluation(results_dir, obj_file, hggNet, keypointGenerator):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    report_data = []

    async def wrapper():
        result = await evaluate_grasps_for_object(obj_file, hggNet, keypointGenerator)

        save_report_to_csv(result, os.path.join(results_dir, 'object_success_rates.csv'))
        report_data.extend(result)

    try:
        loop.run_until_complete(wrapper())
    finally:
        loop.close()

    return report_data


async def main():
    obj_directories = [
        r"E:\YBC",
        r"E:\DexNet",
        r"E:\EGAD\egad_eval_set",
    ]

    # Collect all .obj files from the directories
    obj_files = []
    for directory in obj_directories:
        obj_files.extend(get_obj_files(directory))

    hggNet = HGGQNet().cuda()
    checkpoint = torch.load('HGNet/checkpoints/checkpoint_epoch10.tar', map_location='cpu', weights_only=False)
    hggNet.load_state_dict(checkpoint['model_state_dict'])
    hggNet.eval()

    keypointGenerator = KeypointGeneration(False, True)
    report_data = []

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')
    results_dir = f'results/{current_datetime}'
    os.makedirs(results_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=NUM_WORKER) as executor:
        futures = []
        for obj_file in obj_files:
            future = executor.submit(
                execute_evaluation,
                results_dir,
                obj_file,
                hggNet,
                keypointGenerator
            )
            futures.append(future)

        for future in futures:
            batch_report_data = future.result()
            report_data.extend(batch_report_data)


if __name__ == "__main__":
    asyncio.run(main())
