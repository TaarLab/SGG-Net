import os
import math
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d

from graspnetAPI.grasp import Grasp
from scipy.spatial.transform import Rotation
from graspnetAPI.utils.utils import create_mesh_box

from gs3d.utils.keypoint_utils import distance


def initialize_simulation(client):
    sim = client.require('sim')
    sim.setStepping(False)
    simIK = client.require('simIK')
    return sim, simIK


def setup_copeliasim():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient()
    sim, simIK = initialize_simulation(client)
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
    return sim, gripperHandle, (maxVel, maxAccel, maxJerk), (ikMaxVel, ikMaxAccel, ikMaxJerk), placeConfig, data


def get_points_on_line(x1, y1, x2, y2, n):
    points = []
    for i in range(n):
        t = i / (n - 1)
        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2
        points.append((x, y))
    return points


def create_point_cloud(depth_array, intrinsic_matrix):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    height, width = depth_array.shape

    u = np.linspace(0, width - 1, width)
    v = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(u, v)

    valid_depths = depth_array > 0
    z = depth_array[valid_depths]
    x = (u[valid_depths] - cx) * z / fx
    y = (v[valid_depths] - cy) * z / fy

    # r = rgb_array[:, :, 0][valid_depths]
    # g = rgb_array[:, :, 1][valid_depths]
    # b = rgb_array[:, :, 2][valid_depths]

    point_cloud = np.stack((x, y, z), axis=-1)

    return point_cloud


def get_intrinsic_matrix(resolution, view_angle):
    fx = fy = resolution[0] / (2 * math.tan(view_angle / 2))
    cx = resolution[0] / 2
    cy = resolution[1] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def get_vision_sensor_intrinsic(sim, cam_handler):
    view_angle = sim.getObjectFloatParam(cam_handler, sim.visionfloatparam_perspective_angle)
    image, resolution = sim.getVisionSensorImg(cam_handler)
    return get_intrinsic_matrix(resolution, view_angle)


def read_point_cloud(sim, cam_handler, depth):
    point_cloud = create_point_cloud(depth, get_vision_sensor_intrinsic(sim, cam_handler))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] / 255.0)
    return pcd


def get_pcd_from_cam(sim, cam_name, blank_cam_pcd):
    cam_handler = sim.getObject(cam_name)

    depth, resolution = sim.getVisionSensorDepth(cam_handler, 1)
    depth = np.frombuffer(depth, np.float32)
    # image, resolution = sim.getVisionSensorImg(cam_handler)

    # image = np.frombuffer(image, np.uint8)
    # image.resize([resolution[0], resolution[1], 3])
    depth.resize([resolution[0], resolution[1]])

    pcd = read_point_cloud(sim, cam_handler, depth)

    if blank_cam_pcd is not None:
        origin_handler = sim.getObject('/origin')

        pcd = remove_blank_from_pcd(pcd, blank_cam_pcd)

        origin_from_cam_pos = sim.getObjectPosition(origin_handler, cam_handler)
        origin_from_cam_pos[0] *= -1
        pcd = pcd.translate(np.array(origin_from_cam_pos) * -1)

        cam_ori_from_origin = sim.getObjectOrientation(cam_handler, origin_handler)
        cam_ori_from_origin = sim.alphaBetaGammaToYawPitchRoll(*cam_ori_from_origin)
        cam_ori = Rotation.from_euler('zyx', np.array(cam_ori_from_origin)).inv()
        pcd = pcd.rotate(cam_ori.as_matrix(), center=(0, 0, 0))

    return pcd


def remove_blank_from_pcd(pcd, blank_pcd):
    dist = np.asarray(pcd.compute_point_cloud_distance(blank_pcd))
    return pcd.select_by_index(np.where(dist > 0.0000000001)[0])


def pcd_as_np(pcd):
    point_cloud = np.asarray(pcd.points)
    point_cloud[:, 0] *= -1
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    point_cloud_with_normals = np.concatenate((point_cloud, normals), axis=1)
    return point_cloud_with_normals


def normal_correction(sim, camera_name, pcn):
    cam_handler = sim.getObject(camera_name)
    origin_handler = sim.getObject('/origin')
    cam_from_origin_pos = sim.getObjectPosition(cam_handler, origin_handler)

    pcn = np.array(pcn)

    vectors = pcn[:, :3] - cam_from_origin_pos

    vectors_norm = np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    vectors_normalized = vectors / vectors_norm

    dot_products = np.sum(pcn[:, 3:] * vectors_normalized, axis=1)
    dot_products_negative = np.sum(pcn[:, 3:] * -vectors_normalized, axis=1)

    factors = np.where(dot_products > dot_products_negative, -1, 1)

    return pcn[:, 3:] * factors[:, np.newaxis]


def get_and_save_pcd(sim, camera_name, blank_cam_pcd):
    pcd_cam = get_pcd_from_cam(sim, camera_name, blank_cam_pcd)
    pcn_cam = pcd_as_np(pcd_cam)
    pcn_cam[:, 3:] = normal_correction(sim, camera_name, pcn_cam)
    return pcn_cam


def wait_until_object_stops(sim, object_handler):
    i = 0
    while i < 200:
        time.sleep(0.5)
        linear_velocity, angular_velocity = sim.getObjectVelocity(object_handler)
        if all(abs(v) < 0.01 for v in linear_velocity + angular_velocity):
            break
        i += 1
    time.sleep(0.1)


def get_pcn_from_cameras(sim, cameras, env='/env', port=None):
    env_handler = sim.getObject(env)

    object_handler = None
    while True:
        object_handler = sim.getObjectChild(env_handler, 0)
        if object_handler is None:
            time.sleep(0.2)
        else:
            break

    layer = sim.getIntProperty(object_handler, 'layer')
    sim.setIntProperty(object_handler, 'layer', 0)

    ports = [port] * len(cameras)

    sim.startSimulation()

    blank_point_clouds = []

    def worker(camera_name, port):
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient(port=port)
        sim = client.getObject('sim')
        return get_pcd_from_cam(sim, camera_name, None)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, cameras, ports))

    blank_point_clouds.extend(results)

    time.sleep(1)

    sim.setIntProperty(object_handler, 'layer', layer)

    point_clouds = []

    def worker(camera_name, blank_point_cloud, port):
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient(port=port)
        sim = client.require('sim')
        wait_until_object_stops(sim, object_handler)
        return get_and_save_pcd(sim, camera_name, blank_point_cloud)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, cameras, blank_point_clouds, ports))

    point_clouds.extend(results)

    sim.stopSimulation()

    sim.setIntProperty(object_handler, 'layer', layer)

    return np.concatenate(point_clouds, axis=0)


def load_obj(file_path, desired_height, max_width, load_mode):
    vertices = []
    faces = []
    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                if load_mode == 0:
                    coords = [x, y, -z]
                else:
                    coords = [x, -z, y]
                vertices.append(coords)
                min_coords = [min(min_coords[i], coords[i]) for i in range(3)]
                max_coords = [max(max_coords[i], coords[i]) for i in range(3)]
            elif line.startswith('f '):
                face = [int(idx.split('/')[0]) - 1 for idx in line.split()[1:]]
                faces.extend(face)

    if max_coords[2] - min_coords[2] == 0:
        return None

    scale_factor_height = desired_height / (max_coords[2] - min_coords[2])

    # Calculate width and scale factor based on max width
    current_width = max(abs(max_coords[1] - min_coords[1]), abs(max_coords[0] - min_coords[0]))
    scale_factor_width = max_width / current_width if (current_width * scale_factor_height) > max_width else 1000

    # Use the smaller scale factor to maintain proportions
    scale_factor = min(scale_factor_height, scale_factor_width)

    scaled_vertices = [[coord * scale_factor for coord in vertex] for vertex in vertices]
    scaled_min_max = [[min(vertex[i] for vertex in scaled_vertices), max(vertex[i] for vertex in scaled_vertices)] for i
                      in range(3)]

    if any(scaled_max - scaled_min < 0.003 for scaled_min, scaled_max in scaled_min_max):
        return None

    mid_x = (scaled_min_max[0][0] + scaled_min_max[0][1]) / 2
    mid_y = (scaled_min_max[1][0] + scaled_min_max[1][1]) / 2
    mid_z = (scaled_min_max[2][0] + scaled_min_max[2][1]) / 2

    translated_scaled_vertices = [[vertex[0] - mid_x, vertex[1] - mid_y, vertex[2] - mid_z] for vertex in
                                  scaled_vertices]

    flattened_scaled_vertices = [coord for vertex in translated_scaled_vertices for coord in vertex]
    return flattened_scaled_vertices, faces


def get_obj_files(directory, postfix=''):
    obj_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(postfix + '.obj'):
                obj_files.append(os.path.join(root, file))
    return obj_files


def add_obj_file_to_sim(file_path, client, sim):
    obj = sim.importShape(0, file_path, 0, 0, 1)

    if obj is None:
        return None

    return create_simulation_shapes(client, sim, obj, file_path)


def get_convex_decomposition(sim, shape_handle, morph=False, same=False, use_vhacd=False,
                             individual_meshes=False,
                             hacd_extra_points=True, hacd_face_points=True,
                             hacd_min_clusters=1, hacd_tri_target=500,
                             hacd_max_vertex=200, hacd_max_iter=4,
                             hacd_max_concavity=100, hacd_max_dist=30,
                             hacd_cluster_thresh=0.25,
                             vhacd_pca=False, vhacd_tetrahedron=False,
                             vhacd_res=100000, vhacd_depth=20,
                             vhacd_plane_downsample=4,
                             vhacd_hull_downsample=4,
                             vhacd_max_vertex=64, vhacd_concavity=0.0025,
                             vhacd_alpha=0.05, vhacd_beta=0.05,
                             vhacd_gamma=0.00125, vhacd_min_vol=0.0001):
    """
    Compute the convex decomposition of the specified shape using HACD or V-HACD algorithms.

    :param shape_name: Name or path of the shape to decompose.
    :param morph: Morph the shape into its convex decomposition. Otherwise, create a new shape.
    :param same: Use the same parameters as the last call to the function.
    :param use_vhacd: Use V-HACD algorithm.
    :param individual_meshes: Handle each individual mesh of a compound shape separately during decomposition.
    :param hacd_extra_points: HACD: Add extra points when computing the concavity.
    :param hacd_face_points: HACD: Add face points when computing the concavity.
    :param hacd_min_clusters: HACD: Minimum number of clusters to generate.
    :param hacd_tri_target: HACD: Targeted number of triangles of the decimated mesh.
    :param hacd_max_vertex: HACD: Maximum number of vertices for each generated convex hull.
    :param hacd_max_iter: HACD: Maximum number of iterations.
    :param hacd_max_concavity: HACD: Maximum allowed concavity.
    :param hacd_max_dist: HACD: Maximum allowed distance to get convex clusters connected.
    :param hacd_cluster_thresh: HACD: Threshold to detect small clusters, expressed as a fraction of the total mesh surface.
    :param vhacd_pca: V-HACD: Enable PCA.
    :param vhacd_tetrahedron: V-HACD: Use tetrahedron-based approximate convex decomposition; otherwise, voxel-based decomposition is used.
    :param vhacd_res: V-HACD: Resolution (10000-64000000).
    :param vhacd_depth: V-HACD: Depth (1-32).
    :param vhacd_plane_downsample: V-HACD: Plane downsampling (1-16).
    :param vhacd_hull_downsample: V-HACD: Convex hull downsampling (1-16).
    :param vhacd_max_vertex: V-HACD: Maximum number of vertices per convex hull (4-1024).
    :param vhacd_concavity: V-HACD: Concavity (0.0-1.0).
    :param vhacd_alpha: V-HACD: Alpha (0.0-1.0).
    :param vhacd_beta: V-HACD: Beta (0.0-1.0).
    :param vhacd_gamma: V-HACD: Gamma (0.0-1.0).
    :param vhacd_min_vol: V-HACD: Minimum volume per convex hull (0.0-0.01).
    :return: Handle of the new shape representing the convex decomposition.
    """

    # Set options based on parameters
    options = 0
    if morph:
        options |= 1
    if same:
        options |= 4
    if hacd_extra_points:
        options |= 8
    if hacd_face_points:
        options |= 16
    if individual_meshes:
        options |= 32
    if use_vhacd:
        options |= 128
    if vhacd_pca:
        options |= 256
    if vhacd_tetrahedron:
        options |= 512

    # Integer parameters for the decomposition algorithm
    int_params = [
        hacd_min_clusters,  # [0]
        hacd_tri_target,  # [1]
        hacd_max_vertex,  # [2]
        hacd_max_iter,  # [3]
        0,  # [4]
        vhacd_res,  # [5]
        vhacd_depth,  # [6]
        vhacd_plane_downsample,  # [7]
        vhacd_hull_downsample,  # [8]
        vhacd_max_vertex  # [9]
    ]

    # Float parameters for the decomposition algorithm
    float_params = [
        hacd_max_concavity,  # [0]
        hacd_max_dist,  # [1]
        hacd_cluster_thresh,  # [2]
        0.0,  # [3]
        0.0,  # [4]
        vhacd_concavity,  # [5]
        vhacd_alpha,  # [6]
        vhacd_beta,  # [7]
        vhacd_gamma,  # [8]
        vhacd_min_vol  # [9]
    ]

    return sim.convexDecompose(shape_handle, options, int_params, float_params)


def hacd_decomposition(client, shape_handle, min_cluster_cnt=50, max_concavity=100.0,
                       max_connection_dist=50.0, triangle_cnt_decimated_mesh=1000,
                       max_vertices_cnt=500, small_cluster_detect_threshold=0.25,
                       add_extra_pts=True, add_extra_face_pts=True):
    """
    Perform HACD convex decomposition on the specified shape.

    :param shape_name: Name or path of the shape to decompose.
    :param min_cluster_cnt: Minimum number of clusters to generate.
    :param max_concavity: Maximum allowed concavity.
    :param max_connection_dist: Maximum allowed distance to connect convex clusters.
    :param triangle_cnt_decimated_mesh: Targeted number of triangles for the decimated mesh.
    :param max_vertices_cnt: Maximum number of vertices for each generated convex hull.
    :param small_cluster_detect_threshold: Threshold to detect small clusters (fraction of total mesh surface).
    :param add_extra_pts: Add extra points when computing concavity.
    :param add_extra_face_pts: Add face points when computing concavity.
    :return: List of handles of the new shapes representing the convex decomposition.
    """
    client.require('simConvex')
    sim = client.getObject('sim')
    simConvex = client.getObject('simConvex')

    params = {
        'min_cluster_cnt': min_cluster_cnt,
        'max_concavity': max_concavity,
        'max_connection_dist': max_connection_dist,
        'triangle_cnt_decimated_mesh': triangle_cnt_decimated_mesh,
        'max_vertices_cnt': max_vertices_cnt,
        'small_cluster_detect_threshold': small_cluster_detect_threshold,
        'add_extra_pts': add_extra_pts,
        'add_extra_face_pts': add_extra_face_pts
    }

    convex_shape_handles = sim.groupShapes(simConvex.hacd(shape_handle, params))
    return convex_shape_handles


def vhacd_decomposition(client, shape_handle, resolution=100000, concavity=0.1,
                        plane_downsampling=4, hull_downsampling=4,
                        alpha=0.05, beta=0.05, max_vertices=64,
                        min_volume=0.0001, pca=False, voxels=True):
    """
    Perform V-HACD convex decomposition on the specified shape.

    :param shape_name: Name or path of the shape to decompose.
    :param resolution: Resolution (range: 10000-64000000).
    :param concavity: Concavity (range: 0.0-1.0).
    :param plane_downsampling: Plane downsampling (range: 1-16).
    :param hull_downsampling: Convex hull downsampling (range: 1-16).
    :param alpha: Alpha parameter (range: 0.0-1.0).
    :param beta: Beta parameter (range: 0.0-1.0).
    :param max_vertices: Maximum number of vertices per convex hull (range: 4-1024).
    :param min_volume: Minimum volume per convex hull (range: 0.0-0.01).
    :param pca: Enable Principal Component Analysis (PCA).
    :param voxels: Use voxel-based decomposition; if False, tetrahedron-based decomposition is used.
    :return: List of handles of the new shapes representing the convex decomposition.
    """
    client.require('simConvex')
    simConvex = client.getObject('simConvex')

    params = {
        'resolution': resolution,
        'concavity': concavity,
        'plane_downsampling': plane_downsampling,
        'hull_downsampling': hull_downsampling,
        'alpha': alpha,
        'beta': beta,
        'max_vertices': max_vertices,
        'min_volume': min_volume,
        'pca': pca,
        'voxels': voxels
    }

    convex_shape_handles = simConvex.vhacd(shape_handle, params)
    return convex_shape_handles


def convex_hull(client, shape_handle, growth=0.0):
    """
    Compute the convex hull of one or more objects.

    :param objects: List of names or paths of the objects.
    :param growth: Optional growth parameter.
    :return: Handle of the new shape representing the convex hull.
    """
    client.require('simConvex')
    simConvex = client.getObject('simConvex')

    convex_shape_handle = simConvex.hull([shape_handle], growth)
    return convex_shape_handle


def create_simulation_shapes(client, sim, obj, file_id):
    env_handle = sim.getObject('/env')
    shape_handles = []
    tries = 0
    while tries < 100:
        try:
            tries += 1
            h = obj

            h_convex = h
            sim.relocateShapeFrame(h_convex, [0, 0, 0, 0, 0, 0, 0])
            # h_convex = hacd_decomposition(client, h)
            # sim.removeObjects([h])

            desired_height = 0.08
            max_width = 0.3

            shapeBB = sim.getShapeBB(h_convex)
            bb = shapeBB[0]

            current_height = bb[2]
            current_width = max(abs(bb[0]), abs(bb[1]))

            if current_height > current_width:
                sim.setObjectOrientation(h_convex, [0, math.pi / 2, 0], -1)

                shapeBB = sim.getShapeBB(h_convex)
                bb = shapeBB[0]
                current_height = bb[2]
                current_width = max(abs(bb[0]), abs(bb[1]))

            scale_factor_height = desired_height / current_height
            scale_factor_width = max_width / current_width if (current_width * scale_factor_height) > max_width else 1000

            scale_factor = min(scale_factor_height, scale_factor_width)

            scaled_bb = [dim * scale_factor for dim in bb]

            sim.setShapeBB(h_convex, scaled_bb)

            height = scaled_bb[2]
            sim.setObjectPosition(h_convex, [0, 0, height / 2 + 0.05], -1)

            sim.setShapeColor(h_convex, "", sim.colorcomponent_ambient, [0.5, 0.5, 0.5])
            sim.setObjectInt32Param(h_convex, sim.shapeintparam_respondable, 1)
            sim.setObjectInt32Param(h_convex, sim.shapeintparam_static, 0)
            sim.resetDynamicObject(h_convex)

            sim.setObjectAlias(h_convex, "object_" + str(file_id))
            sim.setObjectParent(h_convex, env_handle, True)

            shape_handles.append(h_convex)
            friction = 1.0
            sim.setFloatProperty(h_convex, 'mass', 0.1)
            sim.setBoolProperty(h_convex, 'bullet.stickyContact', True)
            sim.setBoolProperty(h_convex, 'bullet.autoShrinkConvexMeshes', True)
            sim.setBoolProperty(h_convex, 'bullet.customCollisionMarginEnabled', True)
            sim.setFloatProperty(h_convex, 'bullet.friction', friction)
            sim.setFloatProperty(h_convex, 'bullet.linearDamping', 0.2)
            sim.setFloatProperty(h_convex, 'bullet.angularDamping', 0.2)
            sim.setFloatProperty(h_convex, 'bullet.frictionOld', friction)
            sim.setFloatProperty(h_convex, 'newton.staticFriction', 3)
            sim.setFloatProperty(h_convex, 'newton.kineticFriction', 2.5)
            sim.setFloatProperty(h_convex, 'newton.restitution', 1)
            sim.setFloatProperty(h_convex, 'newton.linearDrag', 0.5)
            sim.setFloatProperty(h_convex, 'newton.angularDrag', 0.5)
            sim.setBoolProperty(h_convex, 'newton.fastMoving', False)
            sim.setFloatProperty(h_convex, 'bullet.customCollisionMarginValue', 0.05)
            sim.setFloatProperty(h_convex, 'bullet.customCollisionMarginConvexValue', 0.001)

            sim.setFloatProperty(sim.handle_scene, 'mujoco.impratio', 50)

            break
        except Exception as e:
            print(f"Failed to load: {file_id} due to {e}")
            cleanup_shapes(sim, shape_handles)
            time.sleep(0.1)
    return shape_handles


def cleanup_shapes(sim, shape_handles):
    try:
        sim.removeObjects(shape_handles)
    except Exception as e:
        print(f"Error during cleanup: {e}")


def setGripperData(sim, gripperHandle, open=True, velocity=None, force=None):
    # sim.callScriptFunction("openClicked" if open else "closeClicked", sim.getObject("/Franka/ROBOTIQ85/Script"))
    if velocity is None:
        velocity = 0.5
    if force is None:
        force = 5 if open else 4
    if not open:
        velocity = -velocity

    sim.setJointTargetForce(gripperHandle, force)
    sim.setJointTargetVelocity(gripperHandle, velocity)


def moveToPoseCallback(sim, simIK, q, velocity, accel, auxData):
    sim.setObjectPose(auxData['target'], sim.handle_world, q)
    simIK.handleGroup(auxData['ikEnv'], auxData['ikGroup'], {'syncWorlds': True})


def moveToPose_viaIK(sim, simIK, maxVelocity, maxAcceleration, maxJerk, targetQ, auxData):
    currentQ = sim.getObjectPose(auxData['tip'], sim.handle_world)
    callback = lambda q, velocity, accel, auxData: moveToPoseCallback(sim, simIK, q, velocity, accel, auxData)
    return sim.moveToPose(-1, currentQ, maxVelocity, maxAcceleration, maxJerk, targetQ, callback, auxData)




def moveToConfig_viaFK(sim, maxVelocity, maxAcceleration, maxJerk, goalConfig, auxData):
    def moveToConfigCallback(config, velocity, accel, auxData):
        for i, jh in enumerate(auxData['joints']):
            if sim.isDynamicallyEnabled(jh):
                sim.setJointTargetPosition(jh, config[i])
            else:
                sim.setJointPosition(jh, config[i])

    startConfig = [sim.getJointPosition(joint) for joint in auxData['joints']]
    sim.moveToConfig(-1, startConfig, None, None, maxVelocity,
                     maxAcceleration, maxJerk, goalConfig, None, moveToConfigCallback, auxData, None)


def vector_to_quaternion(v, perp_vector):
    # Normalize the input vector
    v_norm = v / np.linalg.norm(v)

    # If no reference vector is provided, use the default z-axis
    reference_vector = np.array([-1, 0, 0])

    # Normalize the reference vector
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    # Check if the vectors are nearly opposite
    if np.allclose(v_norm, -reference_vector):
        return np.array([0, 0, 0, -1])  # Special case handling

    # Compute rotation axis
    rotation_axis = cross_product(reference_vector, v_norm)

    # Compute rotation angle (in radians)
    theta = np.arccos(np.dot(reference_vector, v_norm))

    # Convert axis and angle to a quaternion
    q1 = Rotation.from_rotvec(theta * rotation_axis)

    if perp_vector is not None:
        perp_vector /= np.linalg.norm(perp_vector)

        # Determine pitch angle based on how the perpendicular vector deviates from being purely perpendicular to v
        pitch_angle = np.arcsin(np.dot(perp_vector, reference_vector))

        # The rotation axis for the pitch is the same as the input vector v
        q2 = Rotation.from_rotvec(pitch_angle * reference_vector)

        return (q1 * q2).as_quat()
    else:
        return q1.as_quat()


def point_picking_callback(mesh, pid, plotter, _):
    print(pid)
    point = mesh.points[pid]
    dargs = dict(name='labels', font_size=24)
    label = ['ID: {}'.format(pid)]
    plotter.add_point_labels(point, label, **dargs)


def enable_point_picking(plotter, points):
    plotter.enable_point_picking(callback=lambda mesh, pid: point_picking_callback(mesh, pid, plotter, points),
                                 show_message=True,
                                 picker='point', point_size=25,
                                 use_picker=True, show_point=True)


def translate_to_origin(point_cloud, midpoint):
    """Translate point cloud such that midpoint is moved to the origin."""
    return point_cloud - midpoint


def rotation_matrix_to_align_with_target(v, target):
    v = v / np.linalg.norm(v)

    k = target

    r = np.cross(v, k)
    r_norm = np.linalg.norm(r)

    if r_norm == 0:
        if np.dot(v, k) > 0:  # If v is the same as k
            return np.eye(3)
        else:  # v is in the opposite direction of k
            return -np.eye(3)

    r = r / r_norm

    theta = np.arccos(np.dot(v, k))

    # Compute rotation matrix using Rodrigues' formula
    return Rotation.from_rotvec(theta * r)


def cross_product(a, b):
    return np.cross(a, b)


def rotation_matrix_to_align_vectors(start, end, angle):
    axis = np.subtract(end, start)
    axis_normalized = axis / np.linalg.norm(axis)

    return Rotation.from_rotvec(axis_normalized * np.deg2rad(angle))


def check_collision(point1: np.ndarray, point2: np.ndarray, gripper_position: np.ndarray):
    midpoint = np.add(point1, point2) / 2
    normal_plane = cross_product(np.subtract(point1, gripper_position), np.subtract(point2, gripper_position))
    normal_plane = normal_plane / np.linalg.norm(normal_plane)
    normal_plane = rotation_matrix_to_align_vectors(point1, point2, -90).apply(normal_plane)

    grasp_angle = normal_plane

    return grasp_angle * -1, midpoint


def is_above_ground(translation, plane_model, safety_margin=0.005):
    a, b, c, d = plane_model
    x, y, z = translation
    ground_z = (-a * x - b * y - d) / c
    return ground_z - z > safety_margin  # , z - ground_z + safety_margin


def is_fingers_within_workspace(translation, grasp_axis, gripper_span, intrinsic_matrix, img_width, img_height):
    half_span_vector = (gripper_span / 2) * grasp_axis
    right_gripper = np.array([translation[0], translation[1], translation[2]]) + half_span_vector
    left_gripper = np.array([translation[0], translation[1], translation[2]]) - half_span_vector

    def project_to_image_plane(point, intrinsic_matrix):
        x, y, z = point
        projected_point = np.dot(intrinsic_matrix, np.array([x, y, z]))
        u = projected_point[0] / projected_point[2]
        v = projected_point[1] / projected_point[2]
        return u, v

    right_u, right_v = project_to_image_plane(right_gripper, intrinsic_matrix)
    left_u, left_v = project_to_image_plane(left_gripper, intrinsic_matrix)

    margin = 20
    right_in_workspace = margin <= right_u < img_width - margin and margin <= right_v < img_height - margin
    left_in_workspace = margin <= left_u < img_width - margin and margin <= left_v < img_height - margin

    return right_in_workspace and left_in_workspace


def is_fingers_above_ground(translation, plane_model, grasp_axis, gripper_span, safety_margin=0.005):
    a, b, c, d = plane_model
    x_center, y_center, z_center = translation

    half_span_vector = (gripper_span / 2) * grasp_axis

    # Calculate the positions of the right and left grippers
    right_gripper = np.array([x_center, y_center, z_center]) + half_span_vector
    left_gripper = np.array([x_center, y_center, z_center]) - half_span_vector

    # Calculate ground_z for both right and left grippers
    def ground_z_at_position(x, y):
        return (-a * x - b * y - d) / c

    right_gripper_z = ground_z_at_position(right_gripper[0], right_gripper[1])
    left_gripper_z = ground_z_at_position(left_gripper[0], left_gripper[1])

    # Check if both right and left grippers are above the ground
    right_is_above = (right_gripper_z - right_gripper[2]) > safety_margin
    left_is_above = (left_gripper_z - left_gripper[2]) > safety_margin

    # Return if both are above and the minimum margin to ground for further adjustments
    is_above = right_is_above and left_is_above
    margin = max(right_gripper[2] - right_gripper_z, left_gripper[2] - left_gripper_z) + safety_margin

    return is_above, margin


def detect_best_plane(points, min_inliers=12, max_iterations=2, best_plane=None, sample_size=1000):
    def fit_plane_svd(pts):
        centroid = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[2, :]

        # Ensure the normal points upward by checking the Z component
        if normal[2] < 0:
            normal = -normal

        return np.append(normal, -np.dot(normal, centroid))

    def calculate_error(pts, model):
        a, b, c, d = model
        distances = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        return distances

    # Initial plane fitting using the first 'min_inliers' points
    if best_plane is None:
        initial_indices = np.arange(min_inliers)
        plane_model = fit_plane_svd(points[initial_indices])
    else:
        plane_model = best_plane

    # Iteratively refine the plane model
    for _ in range(max_iterations):
        # Deterministically sample points for error calculation
        if len(points) > sample_size:
            step = len(points) // sample_size
            sample_indices = np.arange(0, len(points), step)[:sample_size]
        else:
            sample_indices = np.arange(len(points))
        sampled_points = points[sample_indices]

        # Calculate errors for sampled points
        errors = calculate_error(sampled_points, plane_model)

        # Identify inliers within the sampled points
        if len(errors) // 2 >= 3:
            inlier_indices = sample_indices[np.argsort(errors)[:len(errors) // 2]]
        else:
            inlier_indices = sample_indices

        # If not enough inliers, terminate early
        if len(inlier_indices) < min_inliers:
            break

        # Refine the plane model using the new set of inliers
        plane_model = fit_plane_svd(points[inlier_indices])

    return plane_model


def rotate_point_cloud(pcd, rotation_matrix):
    """
    Apply a rotation matrix to the point cloud.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud.
    - rotation_matrix: numpy.ndarray, the 3x3 rotation matrix to apply.

    Returns:
    - pcd_rotated: open3d.geometry.PointCloud, the rotated point cloud.
    """
    # Get the current point coordinates
    points = np.asarray(pcd.points)

    # Apply the rotation matrix
    rotated_points = np.dot(points, rotation_matrix.T)

    # Create a new point cloud with the rotated points
    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(rotated_points)

    return pcd_rotated


def align_ground_plane(pcd, ground_plane_model):
    """
    Detect the ground plane in the point cloud and rotate the point cloud to align the ground with X and Y axes.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud.
    - distance_threshold: float, maximum distance a point can be from the plane to be considered an inlier.
    - ransac_n: int, number of points to sample for generating a plane model.
    - num_iterations: int, number of iterations to run RANSAC.
    - min_inliers: int, minimum number of inliers to accept a plane.

    Returns:
    - pcd_aligned: open3d.geometry.PointCloud, the point cloud rotated to align the ground plane.
    - ground_plane_model: list of plane equation coefficients [a, b, c, d] for the ground plane.
    """
    # Assuming the ground plane is the largest one or the one with the normal closest to the Z-axis
    # Find the plane whose normal is closest to [0, 0, 1] (the Z-axis)
    z_axis = np.array([0, 0, 1])

    # Compute the rotation matrix to align the ground plane's normal with the Z-axis
    ground_normal = np.array(ground_plane_model[:3])
    ground_normal /= np.linalg.norm(ground_normal)  # Normalize the ground normal vector

    # Find the rotation axis (cross product of ground normal and Z-axis)
    rotation_axis = np.cross(ground_normal, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis

    # Compute the rotation angle
    rotation_angle = np.arccos(np.dot(ground_normal, z_axis))

    if np.isclose(rotation_angle, 0):  # Already aligned
        return pcd

    # Create the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])

    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

    # Rotate the point cloud to align the ground with X and Y axes
    pcd_aligned = rotate_point_cloud(pcd, rotation_matrix)

    return pcd_aligned, rotation_matrix


def rotate_plane_model(plane_model, rotation_matrix):
    # Extract the normal vector and a point on the plane
    normal = np.array(plane_model[:3])
    point_on_plane = -plane_model[3] * normal / np.dot(normal, normal)

    # Rotate the normal vector and the point
    rotated_normal = rotation_matrix @ normal
    rotated_point = rotation_matrix @ point_on_plane

    # Compute the new 'd' coefficient
    d_new = -np.dot(rotated_normal, rotated_point)
    return np.append(rotated_normal, d_new)


def create_gripper_mesh(center, R, width, depth, height=0.004, finger_width=0.004, tail_length=0.04, score=1,
                        color=None):
    depth_base = 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles)
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles)
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles)
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center

    left_mesh = o3d.geometry.TriangleMesh()
    left_mesh.vertices = o3d.utility.Vector3dVector(vertices[:8, :])
    left_mesh.triangles = o3d.utility.Vector3iVector(left_triangles)
    left_mesh.vertex_colors = o3d.utility.Vector3dVector([[color_r, color_g, color_b]] * len(left_points))

    right_mesh = o3d.geometry.TriangleMesh()
    right_mesh.vertices = o3d.utility.Vector3dVector(vertices[8:16, :])
    right_mesh.triangles = o3d.utility.Vector3iVector(right_triangles)
    right_mesh.vertex_colors = o3d.utility.Vector3dVector([[color_r, color_g, color_b]] * len(right_points))

    bottom_mesh = o3d.geometry.TriangleMesh()
    bottom_mesh.vertices = o3d.utility.Vector3dVector(vertices[16:24, :])
    bottom_mesh.triangles = o3d.utility.Vector3iVector(bottom_triangles)
    bottom_mesh.vertex_colors = o3d.utility.Vector3dVector([[color_r, color_g, color_b]] * len(bottom_points))

    tail_mesh = o3d.geometry.TriangleMesh()
    tail_mesh.vertices = o3d.utility.Vector3dVector(vertices[24:32, :])
    tail_mesh.triangles = o3d.utility.Vector3iVector(tail_triangles)
    tail_mesh.vertex_colors = o3d.utility.Vector3dVector([[color_r, color_g, color_b]] * len(tail_points))

    return left_mesh, right_mesh, bottom_mesh, tail_mesh, np.max(vertices[:, 2])


def has_finger_collision(pcd, g: Grasp):
    left_mesh, right_mesh, _, _, _ = create_gripper_mesh(g.translation, g.rotation_matrix,
                                                         g.width, g.depth, 0.01)
    if len(pcd.crop(left_mesh.get_minimal_oriented_bounding_box()).points) > 0:
        return True
    if len(pcd.crop(right_mesh.get_minimal_oriented_bounding_box()).points) > 0:
        return True
    return False


def has_bottom_collision(pcd, g: Grasp):
    _, _, bottom_mesh, tail_mesh, max_z = create_gripper_mesh(g.translation, g.rotation_matrix,
                                                              g.width, g.depth, 0.01)
    if len(pcd.crop(bottom_mesh.get_minimal_oriented_bounding_box()).points) > 0 or len(
            pcd.crop(tail_mesh.get_minimal_oriented_bounding_box()).points) > 0:
        return True
    return False


def find_scored_hands(best_plane, grasp_points, pairing_score, intrinsic_matrix,
                      image_size, gripper_pos=(0, 0, -10), ignore_plane_align=False,
                      depth_translations=[0.03, 0.09], width_translations=[0, 0.005, 0.01, 0.02, 0.03],
                      single_theta=False):
    grasp_point1 = np.array(grasp_points[0])
    grasp_point2 = np.array(grasp_points[1])

    translation = (grasp_point1 + grasp_point2) / 2
    gripper_position = translation + np.array(gripper_pos)

    grasp_axis = grasp_point2 - grasp_point1
    grasp_axis_norm = np.linalg.norm(grasp_axis)
    if grasp_axis_norm == 0.0:
        return []
    grasp_axis = grasp_axis / grasp_axis_norm

    approach_vector = translation - gripper_position
    approach_vector = approach_vector / np.linalg.norm(approach_vector)

    binormal_vector = np.cross(grasp_axis, approach_vector)
    if np.linalg.norm(binormal_vector) == 0:
        return []
    binormal_vector = binormal_vector / np.linalg.norm(binormal_vector)

    approach_vector = np.cross(binormal_vector, grasp_axis)
    approach_vector = approach_vector / np.linalg.norm(approach_vector)

    grasps = []
    safety_margin = 0.01
    thetas = [0.0] if single_theta else np.linspace(-np.pi / 2, np.pi / 2, 5)
    for theta in thetas:
        rotation_about_axis = Rotation.from_rotvec(theta * grasp_axis).as_matrix()

        rotated_approach_vector = rotation_about_axis @ approach_vector
        rotated_binormal_vector = rotation_about_axis @ binormal_vector

        rotation_matrix = np.column_stack((rotated_approach_vector, grasp_axis, rotated_binormal_vector))

        for d_i, depth_translation in enumerate(depth_translations):
            g = Grasp()
            g.translation = translation - (depth_translation * rotated_approach_vector)

            if not ignore_plane_align and not is_above_ground(g.translation - (0.02 * rotated_approach_vector),
                                                              best_plane, 0.02):
                continue

            g.rotation_matrix = rotation_matrix
            grasp_distance = distance(grasp_point2, grasp_point1)
            g.width = grasp_distance
            g.depth = depth_translation
            if d_i != 0:
                g.score /= 10
            g.score = pairing_score
            if theta == 0.0:
                g.score += 0.1

            for i in range(len(width_translations)):
                temp_g = deepcopy(g)
                temp_g.width += width_translations[i]
                temp_g.score += width_translations[i] / 100

                if intrinsic_matrix is not None and not is_fingers_within_workspace(
                        g.translation + (temp_g.depth * rotated_approach_vector), grasp_axis, g.width, intrinsic_matrix,
                        image_size[1], image_size[0]):
                    temp_g.score /= 4

                if not ignore_plane_align:
                    max_tries = 5
                    attempts = 0
                    is_above, margin = is_fingers_above_ground(
                        temp_g.translation + (temp_g.depth * rotated_approach_vector), best_plane,
                        grasp_axis, temp_g.width, safety_margin)
                    while not is_above and attempts < max_tries and abs(margin) > 0.001:
                        plane_normal = np.array(best_plane[:3])
                        delta_translation = -margin * plane_normal / np.linalg.norm(plane_normal)
                        temp_g.translation += delta_translation
                        is_above, margin = is_fingers_above_ground(
                            temp_g.translation + (temp_g.depth * rotated_approach_vector),
                            best_plane, grasp_axis, temp_g.width, safety_margin)
                        attempts += 1
                grasps.append(temp_g)
    return grasps


def find_closest_point(inside_points, grasp_point1):
    distances = np.linalg.norm(inside_points - grasp_point1, axis=1)
    closest_index = np.argmin(distances)
    closest_point = inside_points[closest_index]
    return closest_point


def find_closest_point_normal(inside_points, grasp_point1, normals):
    distances = np.linalg.norm(inside_points - grasp_point1, axis=1)
    closest_index = np.argmin(distances)
    closest_point = inside_points[closest_index]
    closest_normal = normals[closest_index]
    return closest_point, closest_normal


def plot_grasps(combined_cloud, grasps, *args):
    # sss = []
    # for g in grasps:
    #     left_mesh, right_mesh, bottom_mesh, tail_mesh, max_z = create_gripper_mesh(g.translation, g.rotation_matrix,
    #                                                                                g.width, g.depth, 0.01, 0.01)
    #     sss.append(left_mesh)
    #     sss.append(right_mesh)
    #     sss.append(bottom_mesh)
    #     sss.append(tail_mesh)
    o3d.visualization.draw_geometries(
        [combined_cloud, *args] + list(map(lambda x: x.to_open3d_geometry((0, 0, 0)), grasps)))


def get_point_cloud(sim, port=None):
    cameras = [
        '/Vision_sensor_1',
        '/Vision_sensor_2',
        '/Vision_sensor_3',
        '/Vision_sensor_4',
    ]
    return get_pcn_from_cameras(sim, cameras, port=port)


def log_time(start_time, scene_id=None, ann_id=None, mask_index=None, height_index=None, message=""):
    elapsed_time = time.time() - start_time
    prefix = ""

    if scene_id is not None:
        prefix += f"Scene {str(scene_id).replace('scene_', '')} "
    if ann_id is not None:
        prefix += f"Annotation {ann_id} "
    if mask_index is not None:
        prefix += f"Mask {mask_index} "
    if height_index is not None:
        prefix += f"Height Index {height_index} "

    print(f"Time taken: {elapsed_time:.2f} seconds - {prefix:<60} - {message}")


import numpy as np


def plane_transformation(current_plane, target_plane):
    # Extract plane coefficients
    pca, pcb, pcc, pcd = current_plane
    tpa, tpb, tpc, tpd = target_plane

    # Normalize plane normals
    n1 = np.array([pca, pcb, pcc])
    n2 = np.array([tpa, tpb, tpc])
    n1 /= np.linalg.norm(n1) + 1e-10  # Avoid division by zero
    n2 /= np.linalg.norm(n2) + 1e-10

    # Compute orthonormal bases for the planes
    def compute_orthonormal_basis(normal):
        # Choose a vector not parallel to the normal
        base_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(normal, base_vector)
        u /= np.linalg.norm(u) + 1e-10
        v = np.cross(normal, u)
        return u, v

    u1, v1 = compute_orthonormal_basis(n1)
    u2, v2 = compute_orthonormal_basis(n2)

    # Rotation matrix from current plane to target plane
    R1 = np.column_stack((u1, v1, n1))
    R2 = np.column_stack((u2, v2, n2))
    R = R2 @ R1.T

    # Compute plane offsets
    P1 = -pcd * n1 / np.dot(n1, n1)  # Projection of the current plane
    P2 = -tpd * n2 / np.dot(n2, n2)  # Projection of the target plane

    # Translation vector
    t = P2 - R @ P1

    # Assemble the transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T
