import numpy as np
import open3d as o3d
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def convert_point_cloud_to_networkx(point_cloud, dist_th=0.04, node_degree=5, dist_base=False, deg_base=True):
    num_points = point_cloud.shape[0]
    xyz = point_cloud

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

    return graph


def farthest_point_sampling(points, num_points_to_select):
    num_points = points.shape[0]
    selected_indices = []
    selected_points = np.zeros((num_points_to_select, 3))

    # Randomly select the first point
    first_index = np.random.randint(num_points)
    selected_indices.append(first_index)
    selected_points[0] = points[first_index]

    # Calculate distance matrix
    dist_matrix = np.linalg.norm(points - points[first_index], axis=1)

    for i in range(1, num_points_to_select):
        farthest_index = np.argmax(dist_matrix)
        selected_indices.append(farthest_index)
        selected_points[i] = points[farthest_index]

        # Update distance matrix
        dist_matrix = np.minimum(dist_matrix, np.linalg.norm(points - points[farthest_index], axis=1))

    return selected_points


if __name__ == "__main__":
    point_cloud = np.load("gs3d/Visualization/hourglass.npy")[:, :3]

    labels = DBSCAN(eps=0.005, min_samples=10).fit_predict(point_cloud)
    main_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
    point_cloud = point_cloud[labels == main_cluster_label]

    # point_cloud = farthest_point_sampling(point_cloud,100)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    voxel_size = 0.01  # 5 cm voxel resolution  
    point_cloud = np.array(pcd.voxel_down_sample(voxel_size=voxel_size).points)
    G = convert_point_cloud_to_networkx(point_cloud)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i, j in G.edges:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([G.nodes[i]['xyz'], G.nodes[j]['xyz']])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        vis.add_geometry(line)
        node_points = [G.nodes[i]['xyz'] for i in G.nodes]
        node_pcd = o3d.geometry.PointCloud()
        node_pcd.points = o3d.utility.Vector3dVector(node_points)

        # node_pcd.paint_uniform_color([1, 0, 0])  # Set the color of the nodes to red  
        vis.add_geometry(node_pcd)
        
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    K = np.eye(4) 
    K[:3, :3] = np.array([(0.18, 0.18, 0.22),
                               (-0.013, -0.014, 0.028),
                               (0.0, 0.0, 1.0)])
    camera_params.extrinsic = K
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    vis.update_renderer()
    vis.run()

