import numpy as np
import open3d as o3d


def plot_plane_in_open3d(pcd, plane_model, plane_size=1.0):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    plane_center = -d * normal / np.dot(normal, normal)
    plane_x = np.cross([0, 0, 1], normal) if np.linalg.norm(np.cross([0, 0, 1], normal)) > 0 else np.array([1, 0, 0])
    plane_x = plane_x / np.linalg.norm(plane_x)
    plane_y = np.cross(normal, plane_x)
    grid_points = []
    grid_size = np.linspace(-plane_size, plane_size, 10)
    for i in grid_size:
        for j in grid_size:
            grid_points.append(plane_center + i * plane_x + j * plane_y)

    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(grid_points)

    faces = []
    for i in range(len(grid_size) - 1):
        for j in range(len(grid_size) - 1):
            idx = i * len(grid_size) + j
            faces.append([idx, idx + 1, idx + len(grid_size)])
            faces.append([idx + 1, idx + len(grid_size), idx + len(grid_size) + 1])

    plane_mesh.triangles = o3d.utility.Vector3iVector(faces)
    plane_mesh.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd, plane_mesh])
