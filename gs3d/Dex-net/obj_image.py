import os
import numpy as np
import open3d as o3d

from gs3d.Visualization.method import hex_to_rgb_array
from gs3d.utils.utils import get_obj_files

def sample_point_cloud_from_obj(obj_file):
    mesh = o3d.io.read_triangle_mesh(obj_file)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    return mesh


def visualize_and_save_point_cloud(point_cloud, output_image, view_params):
    point_cloud.paint_uniform_color(hex_to_rgb_array('#4e4f50'))

    centroid = np.asarray(point_cloud.get_center())

    view_params['lookat'] = centroid.tolist()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)

    ctr = vis.get_view_control()
    ctr.set_front(view_params['front'])
    ctr.set_lookat(view_params['lookat'])
    ctr.set_up(view_params['up'])
    ctr.set_zoom(view_params['zoom'])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_image)

    vis.destroy_window()

def process_obj_file(obj_file, output_dir, dataset_name, object_index):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    point_cloud = sample_point_cloud_from_obj(obj_file)

    views = [
        {"front": [0.0, 0.0, -1.0], "lookat": [0.0, 0.0, 0.0], "up": [0.0, -1.0, 0.0], "zoom": 0.8},
        {"front": [1.0, 0.0, 0.0], "lookat": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0], "zoom": 0.8},
        {"front": [0.0, -1.0, 0.0], "lookat": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "zoom": 0.8},
        {"front": [-1.0, 0.0, 0.0], "lookat": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0], "zoom": 0.8},
        {"front": [0.0, 1.0, 0.0], "lookat": [0.0, 0.0, 0.0], "up": [0.0, 0.0, -1.0], "zoom": 0.8}
    ]

    for i, view_params in enumerate(views):
        output_image = os.path.join(output_dir, f"{dataset_name}_{object_index}_view{i+1}.png")
        if os.path.exists(output_image):
            continue
        visualize_and_save_point_cloud(point_cloud, output_image, view_params)

    print(f"Saved point cloud visualizations for {obj_file} in {output_dir}")

if __name__ == "__main__":
    obj_directories = [
        r"E:\DexNet",
        r"E:\EGAD\egad_eval_set",
        r"E:\YBC",
    ]

    obj_files = []
    for directory in obj_directories:
        obj_files.extend(get_obj_files(directory))

    output_directory = "results/images"

    for object_index, obj_file_path in enumerate(obj_files):
        try:
            dataset_name = obj_file_path.split(os.sep)[1]
            process_obj_file(obj_file_path, output_directory, dataset_name, object_index)
        except Exception as e:
            print(e)
