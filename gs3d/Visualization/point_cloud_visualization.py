import os
import asyncio

import numpy as np
import pyvista as pv
from alphashape import alphashape
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import open3d as o3d
from graspnetAPI import Grasp, GraspGroup
import random

from gs3d.utils.utils import setup_copeliasim, get_obj_files, add_obj_file_to_sim, get_point_cloud
from gs3d.keypoint_generation import KeypointGeneration
from gs3d.utils.utils import find_closest_point, create_mesh_box, distance, has_bottom_collision, has_finger_collision
from scipy.spatial.transform import Rotation


def create_gripper_mesh(center, R, width, depth, height=0.002, finger_width=0.002, tail_length=0.04, score=1,
                        color=None):
    depth_base = 0.02

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center

    return vertices[0:8, :], vertices[8:16, :], vertices[16:24, :], vertices[24:32, :]


def find_best_gripper_hand(pcd_points, grasp_points):
    grasp_point1 = np.array(grasp_points[0])
    grasp_point2 = np.array(grasp_points[1])

    grasp_point1 = find_closest_point(pcd_points, grasp_point1)
    grasp_point2 = find_closest_point(pcd_points, grasp_point2)

    translation = (grasp_point1 + grasp_point2) / 2
    gripper_position = translation + np.array((0, 0, -10))

    grasp_axis = grasp_point2 - grasp_point1
    grasp_axis_norm = np.linalg.norm(grasp_axis)
    if grasp_axis_norm == 0.0:
        return []
    grasp_axis = grasp_axis / grasp_axis_norm

    approach_vector = translation - gripper_position
    approach_vector = approach_vector / np.linalg.norm(approach_vector)

    binormal_vector = np.cross(grasp_axis, approach_vector)
    binormal_vector = binormal_vector / np.linalg.norm(binormal_vector)

    approach_vector = np.cross(binormal_vector, grasp_axis)
    approach_vector = approach_vector / np.linalg.norm(approach_vector)

    grasps = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

    for theta in np.linspace(0, np.pi, 5):
        rotation_about_axis = Rotation.from_rotvec(theta * grasp_axis).as_matrix()

        rotated_approach_vector = rotation_about_axis @ approach_vector
        rotated_binormal_vector = rotation_about_axis @ binormal_vector

        rotation_matrix = np.column_stack((rotated_approach_vector, grasp_axis, rotated_binormal_vector))
        g = Grasp()
        depth_translation = 0.02
        g.translation = translation - (depth_translation * rotated_approach_vector)
        g.translation += np.array([0, 0, -0.005])
        g.rotation_matrix = rotation_matrix
        grasp_distance = distance(grasp_point2, grasp_point1)
        g.width = grasp_distance + 0.02
        g.depth = depth_translation
        if has_finger_collision(pcd, g) or has_bottom_collision(pcd, g):
            continue
        grasps.append(g)
    return grasps


def rgb_from_array(arr):
    """  
    Create an RGB color specification format from an input array of values between 0 and 1.  
    """
    # Ensure the input array is within the valid range (0 to 1)  
    arr = np.clip(arr, 0, 1)

    # Convert the array values to RGB colors  
    r = arr
    g = 1 - np.abs(arr - 0.5) * 2
    b = 1 - arr

    # Combine the RGB components into a single color specification  
    colors = ['#{:02X}{:02X}{:02X}'.format(int(r[i] * 255), int(g[i] * 255), int(b[i] * 255)) for i in range(len(arr))]

    return colors


def draw_point_cloud(plotter, point_cloud):
    """Draw the 3D point cloud in the given plotter."""
    point_cloud_pv = pv.PolyData(point_cloud)
    z = point_cloud[:, 2]
    z = (z - np.min(z)) / np.max(z)
    # cmap = ListedColormap(z,'jet').colors
    # cmap = rgb_from_array(cmap)
    point_cloud_pv.point_data['mamad'] = z

    plotter.add_points(
        point_cloud_pv,
        cmap='jet',
        point_size=5,
        render_points_as_spheres=True,
        opacity=0.2,
        lighting=True,
        ambient=0.4,
        specular=0.3,
        scalars='mamad',
        smooth_shading=True
    )
    plotter.remove_scalar_bar()


def save_individual_plot(view_name, plot_function, point_cloud, normals, img_save_path):
    """Helper function to save individual plots"""
    # Create a new plotter for each individual view and enable off-screen rendering
    individual_plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    # individual_plotter.remove_scalar_bar()
    plot_function(individual_plotter, point_cloud, normals)
    # individual_plotter.screenshot(f"{view_name}.png")
    individual_plotter.save_graphic(os.path.join(img_save_path, f"{view_name}.svg"))
    individual_plotter.close()


def visualize_fullscreen_with_four_modes(point_cloud, normals, img_save_path=None, save=False):
    # Create a plotter with full-screen and six viewports (three on top, three at the bottom), and enable off-screen rendering
    plotter = pv.Plotter(shape=(2, 3), title="3 Modes of Visualization with Stacked Maps and Polygons",
                         window_size=[2000, 1200], off_screen=save)
    if save and img_save_path == None:
        print("specify path if you want to save")
        return

        # Row 1: Three columns for point cloud views
    # Plot the point cloud in the first viewport (top-left)
    plotter.subplot(0, 0)
    draw_point_cloud(plotter, point_cloud)
    plotter.add_text("Point Cloud", font_size=12)

    if save:
        save_individual_plot("point_cloud_plot", draw_point_cloud, point_cloud, img_save_path)

    plotter.camera_position = [(0.18, 0.18, 0.22),
                               (-0.013, -0.014, 0.028),
                               (0.0, 0.0, 1.0)]

    # Plot the point cloud with cross-section planes in the second viewport (top-center)
    plotter.subplot(0, 1)
    draw_point_cloud(plotter, point_cloud)
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    add_cross_sections(plotter, z_min, z_max, x_min, x_max, y_min, y_max)
    plotter.add_text("Point Cloud with Cross Sections", font_size=12)

    if save:
        save_individual_plot("cross_section_plot", lambda p, pc: (
        draw_point_cloud(p, pc), add_cross_sections(p, z_min, z_max, x_min, x_max, y_min, y_max)), point_cloud,
                             img_save_path)

    # Plot the flattened 2D maps in the third viewport (top-right)
    plotter.subplot(0, 2)
    # visualize_flattened_sections(plotter, point_cloud)
    # plotter.add_text("Flattened Sections", font_size=12)

    # if save:
    #     save_individual_plot("flattened_sections_plot", visualize_flattened_sections, point_cloud,img_save_path)
    visualize_selected_gripper(plotter, point_cloud, normals, 0.03)
    if save:
        save_individual_plot("Selected Grasps", visualize_selected_gripper, point_cloud, normals, img_save_path)

    # Row 2: Two columns, stacked maps in the left and polygon visualization in the right
    plotter.subplot(1, 0)
    visualize_gripper(plotter, point_cloud, )
    plotter.camera_position = [(0.18, 0.18, 0.22),
                               (-0.013, -0.014, 0.028),
                               (0.0, 0.0, 1.0)]
    plotter.add_text("Gripper Visualisaztion", font_size=12)

    if save:
        save_individual_plot("Gripper Visualisaztion", visualize_gripper, point_cloud, img_save_path)

    plotter.subplot(1, 1)
    visualize_polygons_from_flattened_map(plotter, point_cloud)
    plotter.add_text("Polygon Visualization", font_size=12)

    if save:
        save_individual_plot("polygon_visualization_plot", visualize_polygons_from_flattened_map, point_cloud,
                             img_save_path)

    # Plot the polygon with grasp visualization in the third column (bottom-right)
    plotter.subplot(1, 2)
    visualize_polygons_from_flattened_map(plotter, point_cloud)
    visualize_grasp_pairs(plotter, point_cloud)
    plotter.add_text("Polygon with Grasp Visualization", font_size=12)

    if save:
        save_individual_plot("polygon_grasp_plot",
                             lambda p, pc: (visualize_polygons_from_flattened_map(p, pc), visualize_grasp_pairs(p, pc)),
                             point_cloud, img_save_path)

    # Show the full-screen plot with six viewports
    plotter.show()


def add_cross_sections(plotter, z_min, z_max, x_min, x_max, y_min, y_max, step_size=0.03):
    margin = 0.01  # Extra size (in units) for the plane compared to the object
    z_positions = np.arange(z_min, z_max, step_size)
    for z in z_positions:
        plane = pv.Plane(center=(0, 0, z),
                         direction=(0, 0, 1),
                         i_size=(x_max - x_min + margin),  # Plane size along X
                         j_size=(y_max - y_min + margin))  # Plane size along Y
        plotter.add_mesh(plane, color="orange", opacity=0.3)

    def print_camera_position(_):
        print("Camera Position:", plotter.camera_position)

    plotter.track_click_position(print_camera_position)

    plotter.camera_position = [(0.18, 0.18, 0.22),
                               (-0.013, -0.014, 0.028),
                               (0.0, 0.0, 1.0)]


def visualize_flattened_sections(plotter, point_cloud, step_size=0.03):
    keypointGenerator = KeypointGeneration(False, True)
    selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(point_cloud, step_size)

    # For each Z section, project it into a 2D plane and align them at the lower Z bound
    all_points = np.empty((0, 3))
    for section, height in zip(selected_points, cross_heights):
        if section.shape[0] == 0:
            continue  # Skip if no points in this section

        section_2d = section[:, :2]

        # Flatten to 2D (X, Y) and set Z to the lower boundary of the section (z)
        section_2d = np.c_[section_2d, np.full(section_2d.shape[0], height)]  # Set Z to the lower boundary (z)
        all_points = np.concatenate((all_points, section_2d), axis=0)

    # Add the section to the plotter
    draw_point_cloud(plotter, all_points)

    def print_camera_position(_):
        print("Camera Position:", plotter.camera_position)

    plotter.track_click_position(print_camera_position)

    plotter.camera_position = [(0.18, 0.18, 0.22),
                               (-0.013, -0.014, 0.028),
                               (0.0, 0.0, 1.0)]


def visualize_stacked_flattened_maps(plotter, point_cloud, step_size=0.03, height_offset=0.01, xy_margin=0.01):
    """Visualize flattened maps with vertical offset and X, Y margins."""
    # Get Z bounds and divide into sections
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    z_positions = np.arange(z_min, z_max, step_size)

    # For each Z section, project it into a 2D plane and align them at the lower Z bound
    for i, z in enumerate(z_positions):
        # Select points in the current section
        section = point_cloud[(point_cloud[:, 2] >= z) & (point_cloud[:, 2] < z + step_size)]

        if section.shape[0] == 0:
            continue  # Skip if no points in this section

        # Flatten to 2D (X, Y), apply a margin for X and Y, and stack them vertically
        section_2d = section[:, :2]
        section_2d[:, 0] += i * xy_margin  # Apply X margin
        section_2d[:, 1] += i * -xy_margin  # Apply Y margin
        section_2d = np.c_[section_2d, np.full(section_2d.shape[0], i * height_offset)]  # Apply Z offset (stack height)

        # Add the section to the plotter
        draw_point_cloud(plotter, section_2d)

    def print_camera_position(_):
        print("Camera Position:", plotter.camera_position)

    plotter.track_click_position(print_camera_position)

    plotter.camera_position = [(0.20578472518648397, 0.16770900433682628, 0.07330678631892118),
                               (0.015040039547903675, -0.014977882006496343, 0.015),
                               (-0.14920689719759958, -0.15585673358538013, 0.9764455849788871)]


def visualize_polygons_from_flattened_map(plotter, point_cloud, step_size=0.03):
    keypointGenerator = KeypointGeneration(False, True)
    selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(point_cloud, step_size)

    # For each Z section, project it into a 2D plane and align them at the lower Z bound
    for section, height in zip(selected_points, cross_heights):
        if section.shape[0] == 0:
            continue  # Skip if no points in this section

        labels = DBSCAN(eps=0.005, min_samples=10).fit_predict(section)
        main_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
        section = section[labels == main_cluster_label]
        section_2d = section[:, :2]

        scaler = MinMaxScaler()
        # Scale the points
        scaled_2d_points = scaler.fit_transform(section_2d)

        # Compute alpha shape
        alpha_shape = alphashape(scaled_2d_points, 0.)
        if alpha_shape.geom_type == 'Polygon':
            shape_poly = [alpha_shape]
            sorted_points = np.array(alpha_shape.exterior.coords)
            sorted_points = scaler.inverse_transform(sorted_points)
        else:
            shape_poly = alpha_shape.geoms
            sorted_points = np.array(alpha_shape.geoms[0].exterior.coords)
            sorted_points = scaler.inverse_transform(sorted_points)

        # Create 3D polygon (by adding a constant Z)
        sorted_points = np.c_[sorted_points, np.full(sorted_points.shape[0], height)]  # Add Z value to make it 3D

        # Define the faces of the polygon
        n_points = len(sorted_points)
        faces = np.hstack([[n_points] + list(range(n_points))])  # Face description

        # Create a PolyData object for the polygon
        poly_pv = pv.PolyData(sorted_points, faces)

        # Add the polygon mesh to the plotter
        plotter.add_mesh(poly_pv, color="orange", opacity=0.6)

    def print_camera_position(_):
        print("Camera Position:", plotter.camera_position)

    plotter.track_click_position(print_camera_position)

    plotter.camera_position = [(0.18, 0.18, 0.22),
                               (-0.013, -0.014, 0.028),
                               (0.0, 0.0, 1.0)]


def visualize_selected_gripper(plotter, point_cloud, normals, step_size=0.03):
    point_cloud_pv = pv.PolyData(point_cloud)
    z = point_cloud[:, 2]
    z = (z - np.min(z)) / np.max(z)
    # cmap = ListedColormap(z,'jet').colors
    # cmap = rgb_from_array(cmap)
    point_cloud_pv.point_data['mamad'] = z

    plotter.add_points(
        point_cloud_pv,
        cmap='jet',
        point_size=5,
        render_points_as_spheres=True,
        opacity=0.2,
        lighting=True,
        ambient=0.4,
        specular=0.3,
        scalars='mamad',
        smooth_shading=True
    )
    plotter.remove_scalar_bar()
    grasp_points = []
    skeleton_points_3d = []
    keypointGenerator = KeypointGeneration(False, True)
    selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(point_cloud, step_size)
    indices = np.arange(len(cross_heights))

    for height_index in indices:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        finalGraspPoses, scores, skeleton_points = keypointGenerator.generate_scored_grasp_pairs(height_index, pcd, point_cloud, step_size)
        skeleton_points_3d += skeleton_points

        grasp_points = finalGraspPoses

    grasps = []
    for grasp in grasp_points:
        grasps += find_best_gripper_hand(point_cloud, grasp)

    gg = GraspGroup()
    for g in grasps:
        gg.add(g)

    gg = gg.nms()
    boxes = []
    for g in gg:
        r = random.random()
        if r < 0.6:
            continue
        left_v, right_v, bottom_v, tail_v = create_gripper_mesh(g.translation, g.rotation_matrix, g.width, g.depth)
        bottom_b = pv.Box()
        bottom_b.points = bottom_v
        left_b = pv.Box()
        left_b.points = left_v
        right_b = pv.Box()
        right_b.points = right_v
        tail_b = pv.Box()
        tail_b.points = tail_v
        boxes.append([left_b, right_b, bottom_b, tail_b])

    for box in boxes:
        for i in range(4):
            plotter.add_mesh(box[i], color=[0.7, 0.7, 0.7])


def visualize_gripper(plotter, point_cloud, step_size=0.03, random_sample=False):
    point_cloud_pv = pv.PolyData(point_cloud)
    z = point_cloud[:, 2]
    z = (z - np.min(z)) / np.max(z)
    # cmap = ListedColormap(z,'jet').colors
    # cmap = rgb_from_array(cmap)
    point_cloud_pv.point_data['mamad'] = z

    plotter.add_points(
        point_cloud_pv,
        cmap='jet',
        point_size=5,
        render_points_as_spheres=True,
        opacity=0.2,
        lighting=True,
        ambient=0.4,
        specular=0.3,
        scalars='mamad',
        smooth_shading=True
    )
    plotter.remove_scalar_bar()
    grasp_points = []
    skeleton_points_3d = []
    keypointGenerator = KeypointGeneration(False, True)
    selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(point_cloud, step_size)
    indices = np.arange(len(cross_heights))

    for height_index in indices:
        height = cross_heights[height_index]

        finalGraspPoses, skeleton_points = keypointGenerator.generate_grasp_poses(height_index, point_cloud, step_size,
                                                                                  True)
        skeleton_points_3d += skeleton_points

        if finalGraspPoses is None:
            continue

        grasp_points += [((x1, y1, height), (x2, y2, height)) for ((x1, y1), (x2, y2)) in finalGraspPoses]

    grasps = []
    for grasp in grasp_points:
        grasps += find_best_gripper_hand(point_cloud, grasp)

    gg = GraspGroup()
    for g in grasps:
        gg.add(g)

    gg = gg.nms()
    boxes = []
    for g in gg:
        if (random_sample):
            r = random.random()
            if r < 0.6:
                continue
        left_v, right_v, bottom_v, tail_v = create_gripper_mesh(g.translation, g.rotation_matrix, g.width, g.depth)
        bottom_b = pv.Box()
        bottom_b.points = bottom_v
        left_b = pv.Box()
        left_b.points = left_v
        right_b = pv.Box()
        right_b.points = right_v
        tail_b = pv.Box()
        tail_b.points = tail_v
        boxes.append([left_b, right_b, bottom_b, tail_b])

    for box in boxes:
        for i in range(4):
            plotter.add_mesh(box[i], color=[0.7, 0.7, 0.7])


def visualize_grasp_pairs(plotter, point_cloud, step_size=0.03):
    grasp_points = []
    skeleton_points_3d = []
    keypointGenerator = KeypointGeneration(False, True)
    selected_points, cross_heights = keypointGenerator.split_point_cloud_by_height(point_cloud, step_size)
    indices = np.arange(len(cross_heights))

    for height_index in indices:
        height = cross_heights[height_index]

        finalGraspPoses, skeleton_points = keypointGenerator.generate_grasp_poses(height_index, point_cloud, step_size,
                                                                                  True)
        skeleton_points_3d += skeleton_points

        if finalGraspPoses is None:
            continue

        grasp_points += [((x1, y1, height), (x2, y2, height)) for ((x1, y1), (x2, y2)) in finalGraspPoses]

    for (p1, p2) in grasp_points:
        # Convert grasp pairs into numpy arrays for easier plotting
        p1 = np.array(p1)
        p2 = np.array(p2)

        # Plot the two grasp points and the line connecting them
        plotter.add_mesh(pv.Line(p1, p2), color='blue', line_width=4)  # Line connecting grasp points
        plotter.add_mesh(pv.Sphere(radius=0.001, center=p1), color='red')  # Sphere at p1
        plotter.add_mesh(pv.Sphere(radius=0.001, center=p2), color='red')  # Sphere at p2


async def main():
    point_cloud_file = 'hourglass.npy'
    img_save_path = "E:/TaarLab/3D-Skeleton/graphical_abstract"
    if os.path.exists(point_cloud_file):
        print(f"Loading point cloud from {point_cloud_file}...")
        point_cloud = np.load(point_cloud_file)
        normals = point_cloud[:, 3:]
        point_cloud = point_cloud[:, :3]
    else:
        sim, gripperHandle, forward_config, inverse_config, placeConfig, data = setup_copeliasim()
        sim.stopSimulation()
        obj_directory = r"C:\Users\ARB\PycharmProjects\GraspSkeleton3D\gs3d\Visualization"
        obj_files = get_obj_files(obj_directory)

        for obj_file in obj_files:
            while True:
                try:
                    add_obj_file_to_sim(obj_file, obj_file, obj_directory, sim, 1)
                    point_cloud = get_point_cloud(sim)[:, :3]
                    break
                except Exception as e:
                    print(e)
        labels = DBSCAN(eps=0.005, min_samples=10).fit_predict(point_cloud)
        main_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
        point_cloud = point_cloud[labels == main_cluster_label]

        print(f"Saving point cloud to {point_cloud_file}...")
        np.save(point_cloud_file, point_cloud)

    # Visualize all four modes in full screen
    visualize_fullscreen_with_four_modes(point_cloud, normals, img_save_path, True)


if __name__ == "__main__":
    asyncio.run(main())
