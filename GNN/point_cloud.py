from graspnetAPI.grasp import Grasp, GraspGroup
from graspnetAPI.graspnet import GraspNet
from graspnetAPI.utils.eval_utils import get_scene_name

from GNN.model import RewardNetwork
from sklearn.neighbors import NearestNeighbors
import torch
import networkx as nx
import os
import open3d as o3d
import numpy as np

def pc_to_graph(pcd,dist_th=0.04, node_degree=5, dist_base=False, deg_base=True):
    all_node_features = []
    all_edge_indices = []

    xyz = np.asarray(pcd.points)

    num_points = xyz.shape[0]

    graph = nx.Graph()
    for i in range(num_points):
        node_features = {'xyz': xyz[i]}
        graph.add_node(i, **node_features)

    # Add edges based on distance threshold
    if dist_base:
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance = np.linalg.norm(xyz[i] - xyz[j])
                if distance <= dist_th:
                    graph.add_edge(i, j)

    # Add edges based on k-nearest neighbors
    if deg_base:
        kdtree = NearestNeighbors(n_neighbors=node_degree + 1, algorithm='kd_tree')
        kdtree.fit(xyz)
        for i, point in enumerate(xyz):
            _, neighbor_indices = kdtree.kneighbors([point], node_degree + 1)
            for neighbor_index in neighbor_indices[0]:
                if i != neighbor_index:
                    graph.add_edge(i, neighbor_index)

    # Extract node features and edge index for this point cloud
    node_features = np.array([data['xyz'] for _, data in graph.nodes(data=True)])
    edge_index = np.array(list(graph.edges)).T  # Transpose to get shape [2, num_edges]


    return node_features,edge_index

    


def find_points_in_gripper(gripper, point_cloud):  
    """  
    Find all points of a point cloud that are inside the closing volume of a gripper.  

    Args:  
        gripper (o3d.geometry.TriangleMesh): The gripper mesh.  
        point_cloud (o3d.geometry.PointCloud): The point cloud.  

    Returns:  
        np.ndarray: The indices of the points that are inside the gripper's closing volume.  
    """  
    gripper_obb = gripper.get_oriented_bounding_box() 
    inside_indices = gripper_obb.get_point_indices_within_bounding_box(point_cloud.points)
    inliers_pcd = point_cloud.select_by_index(inside_indices, invert=False)
    return inliers_pcd

if __name__ == "__main__":
    dump_folder_path = "E:/TaarLab/3D-Skeleton/dump_folder"
    camera = "realsense"
    grasp_net_root = "E:/TaarLab/downloaded_datasets/train_4"
    save_dir = "E:/TaarLab/3D-Skeleton/saved_scores"  
    scene_id = 130
    grasp_net = GraspNet(grasp_net_root, camera=camera, split='custom')

    for ann_id in range(256):
        original_pcd = grasp_net.loadScenePointCloud(scene_id, camera, ann_id)
        original_pcd = original_pcd.voxel_down_sample(0.005)
        data_path = os.path.join(save_dir,get_scene_name(scene_id),"annotation_%04d"%ann_id)
        grasp_group = GraspGroup().from_npy(os.path.join(data_path, 'grasp_list.npy'))
        for i in range(len(grasp_group.grasp_group_array)):
            g = Grasp(grasp_group.grasp_group_array[i])
            gripper = g.to_incomplete_gripper()
            o3d.visualization.draw_geometries([original_pcd,gripper])
            inliers_pcd = find_points_in_gripper(gripper,original_pcd)
            o3d.visualization.draw_geometries([inliers_pcd,gripper])
            # graph = pc_to_graph(inliers_pcd)
            # node_features = torch.tensor(node_features, dtype=torch.float32).to("cpu")
            # edge_index = torch.tensor(edge_index).to("cpu")
            # predicted_reward = reward_network(node_features, edge_index, batch_index, action_tensor)
            # predicted_reward = predicted_reward.item()
            



