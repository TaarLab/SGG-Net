import numpy as np
import open3d as o3d
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def pc_to_graph(point_cloud, dist_th=0.04, node_degree=5, dist_base=False, deg_base=True):
    all_node_features = []
    all_edge_indices = []
    batch_index = []

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    xyz = np.asarray(point_cloud_o3d.points)

    num_points = point_cloud.shape[0]

    graph = nx.Graph()
    for i in range(num_points):
        node_features = {'xyz': xyz[i]}
        graph.add_node(i, **node_features)

    if dist_base:
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance = np.linalg.norm(xyz[i] - xyz[j])
                if distance <= dist_th:
                    graph.add_edge(i, j)

    if deg_base:
        kdtree = NearestNeighbors(n_neighbors=node_degree + 1, algorithm='kd_tree')
        kdtree.fit(xyz)
        for i, point in enumerate(xyz):
            _, neighbor_indices = kdtree.kneighbors([point], node_degree + 1)
            for neighbor_index in neighbor_indices[0]:
                if i != neighbor_index:
                    graph.add_edge(i, neighbor_index)

    node_features = np.array([data['xyz'] for _, data in graph.nodes(data=True)])
    edge_index = np.array(list(graph.edges)).T

    all_node_features.append(node_features)
    all_edge_indices.append(edge_index)

    batch_index.append(np.full(node_features.shape[0], 0))

    all_node_features = np.concatenate(all_node_features, axis=0)
    all_edge_indices = np.concatenate(all_edge_indices, axis=1)
    batch_index = np.concatenate(batch_index, axis=0)

    return all_node_features, all_edge_indices, batch_index

def pc_to_graph_from_zarr(zarr_store, cache, pc_keys, dist_th=0.04, node_degree=5, dist_base=False, deg_base=True):
    all_node_features = []
    all_edge_indices = []
    batch_index = []

    for batch_idx, pc_key in enumerate(pc_keys):
        cached_result = cache.get(pc_key) if cache is not None else None
        if cached_result is not None:
            node_features, edge_index = cached_result
        else:
            point_cloud = zarr_store[pc_key][:]

            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

            # point_cloud_o3d = point_cloud_o3d.voxel_down_sample(0.01)

            xyz = np.asarray(point_cloud_o3d.points)
            # xyz = point_cloud[:, :3]

            num_points = point_cloud.shape[0]

            # Create a graph and add nodes with features xyz
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

            if cache is not None:
                cache.put(pc_key, (node_features, edge_index))

        # Append to lists
        all_node_features.append(node_features)
        all_edge_indices.append(edge_index)

        # Update batch_index for each point cloud in the batch
        batch_index.append(np.full(node_features.shape[0], batch_idx))

    # Convert lists to numpy arrays and concatenate
    all_node_features = np.concatenate(all_node_features, axis=0)
    all_edge_indices = np.concatenate(all_edge_indices, axis=1)
    batch_index = np.concatenate(batch_index, axis=0)

    return all_node_features, all_edge_indices, batch_index


def pc_to_graph_from_graspnet(cache, grasp_net, pc_keys, dist_th=0.04, node_degree=5, dist_base=False, deg_base=True):
    all_node_features = []
    all_edge_indices = []
    batch_index = []

    for batch_idx, pc_key in enumerate(pc_keys):
        cached_result = cache.get(pc_key)
        if cached_result is not None:
            node_features, edge_index = cached_result
        else:
            pc_info = pc_key.split('-')
            pcd = grasp_net.loadScenePointCloud(int(pc_info[0]), "kinect", int(pc_info[1]), format='open3d', use_inpainting=True)
            point_cloud = pcd.voxel_down_sample(0.01)
            xyz = np.asarray(point_cloud.points)
            num_points = xyz.shape[0]

            # Create a graph and add nodes with features xyz
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

            cache.put(pc_key, (node_features, edge_index))

        # Append to lists
        all_node_features.append(node_features)
        all_edge_indices.append(edge_index)

        # Update batch_index for each point cloud in the batch
        batch_index.append(np.full(node_features.shape[0], batch_idx))

    # Convert lists to numpy arrays and concatenate
    all_node_features = np.concatenate(all_node_features, axis=0)
    all_edge_indices = np.concatenate(all_edge_indices, axis=1)
    batch_index = np.concatenate(batch_index, axis=0)

    return all_node_features, all_edge_indices, batch_index