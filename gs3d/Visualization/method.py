import os
import numpy as np
import skgeom as sg
import pyvista as pv
from alphashape import alphashape
from matplotlib import pyplot as plt
from shapely.geometry.point import Point
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry.linestring import LineString


def hex_to_rgb_array(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def scale_mesh(mesh, scale_factor):
    scaled_mesh = mesh.copy()
    scaled_mesh = scaled_mesh.scale([scale_factor, scale_factor, scale_factor])
    return scaled_mesh


def repair_and_voxelize(mesh, density=0.01):
    repaired_mesh = mesh.clean()

    voxelized = pv.voxelize(repaired_mesh, density=density, check_surface=False)
    return voxelized


def sample_point_cloud_from_mesh(obj_file, color, density=0.08, scale_factor=1.0):
    try:
        mesh = pv.read(obj_file)
    except Exception as e:
        raise RuntimeError(f"Error reading mesh from file: {obj_file}. Error: {e}")

    scaled_mesh = scale_mesh(mesh, scale_factor)

    voxelized = repair_and_voxelize(scaled_mesh, density)

    point_cloud = voxelized.cell_centers().points

    point_cloud_polydata = pv.PolyData(point_cloud)

    point_cloud_polydata["Colors"] = np.tile(color, (point_cloud_polydata.n_points, 1))
    return point_cloud_polydata


def split_point_cloud(point_cloud, colors):
    points = point_cloud.points

    z_coords = points[:, 1]

    sorted_indices = np.argsort(z_coords)

    split_indices = np.array_split(sorted_indices, len(colors))
    split_clouds = []

    for indices, hex_color in zip(split_indices, colors):
        sub_cloud_points = points[indices]

        sub_cloud = pv.PolyData(sub_cloud_points)

        color_array = np.tile(hex_color, (len(indices), 1))
        sub_cloud["Colors"] = color_array

        split_clouds.append(sub_cloud)

    return split_clouds


def visualize_and_save_point_clouds(clouds, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)

    for i, cloud in enumerate(clouds):
        plotter = pv.Plotter(lighting="light_kit", off_screen=True)
        plotter.enable_depth_peeling()
        plotter.clear()
        plotter.add_mesh(cloud,
                         scalars='Colors',
                         rgb=True,
                         point_size=5,
                         render_points_as_spheres=True,
                         lighting=True,
                         ambient=0.4,
                         specular=0.3,
                         smooth_shading=True
                         )

        plane = pv.Plane(center=(0, 0, -0.1), i_size=10, j_size=10)
        plotter.add_mesh(plane, color='white', opacity=0.5)

        # Don't remove it
        # def print_camera_position(_):
        #     print("Camera Position:", plotter.camera_position)
        # plotter.track_click_position(print_camera_position)

        plotter.camera_position = [(-14.72563820212639, 1.2312697834334705, -16.003906806065096),
                                   (-14.881514316376428, 1.944864587845622, -7.67068461272462),
                                   (-0.00010408811382345371, 0.9963534069707393, -0.08532219865583247)]

        output_image = os.path.join(output_dir, f"{base_name}_split_part{i + 1}.png")
        plotter.show(screenshot=output_image)
        print(f"Saved visualization to {output_image}")


def simplify_polygon(polygon, max_iterations=25):
    if polygon.is_simple():
        return polygon

    new_poly = polygon
    iterations = 0

    while not new_poly.is_simple() and iterations < max_iterations and len(new_poly) > 2:
        new_poly = sg.simplify(new_poly, 0.1, "ratio")
        iterations += 1

    return new_poly


def apply_straight_skeleton_to_each_split(split_clouds, output_dir):
    for i, split_cloud in enumerate(split_clouds):
        points_2d = np.unique(split_cloud.points[:, [0, 2]], axis=0)
        scaler = MinMaxScaler()
        scaled_2d_points = scaler.fit_transform(points_2d)

        alpha = 9.0
        alpha_shape = alphashape(scaled_2d_points, alpha)
        while not alpha_shape.geom_type == 'Polygon' and alpha >= 0.0:
            alpha_shape = alphashape(scaled_2d_points, alpha)
            alpha -= 1.0

        if not alpha_shape.geom_type == 'Polygon':
            print(f"Failed to create a valid alpha shape polygon for split {i + 1}.")
            continue

        sorted_points = np.array(alpha_shape.exterior.coords)
        sorted_points = scaler.inverse_transform(sorted_points)

        polygon_points = sorted_points
        polygon = sg.Polygon(np.array(polygon_points)[::-1])

        polygon = simplify_polygon(polygon)
        polygon_points = np.array([np.array([p.x(), p.y()], dtype=np.float32) for p in list(polygon.vertices)])

        straight_skeleton = sg.skeleton.create_interior_straight_skeleton(polygon)

        if not straight_skeleton:
            print(f"Failed to generate a valid straight skeleton for split {i + 1}.")
            continue

        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')

        if hasattr(polygon, 'holes') and polygon.holes:
            outer_boundary = np.array([(v.x(), v.y()) for v in polygon.outer_boundary().vertices])
            plt.fill(outer_boundary[:, 0], outer_boundary[:, 1], color='lightgray', alpha=0.5, label='Polygon')

            for hole in polygon.holes:
                hole_boundary = np.array([(v.x(), v.y()) for v in hole.vertices])
                plt.fill(hole_boundary[:, 0], hole_boundary[:, 1], color='white')
        else:
            outer_boundary = np.array([(v.x(), v.y()) for v in polygon.vertices])
            plt.fill(outer_boundary[:, 0], outer_boundary[:, 1], color='lightgray', alpha=0.5, label='Polygon')

        def is_on_border(point, polygon_points, threshold=0.05):
            boundary = LineString(polygon_points)
            shapely_point = Point(point.x(), point.y())
            return boundary.distance(shapely_point) < threshold

        def nearest_distance_to_border(point, polygon_points):
            boundary = LineString(polygon_points)
            shapely_point = Point(point.x(), point.y())
            return boundary.distance(shapely_point)

        for halfedge in straight_skeleton.halfedges:
            if halfedge.is_bisector:
                p1 = halfedge.vertex.point
                p2 = halfedge.opposite.vertex.point

                plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], color='#E7625F', linewidth=1)

                if is_on_border(p1, polygon_points):
                    plt.scatter(p1.x(), p1.y(), color='#0e86d4', s=50, zorder=5)
                else:
                    radius = nearest_distance_to_border(p1, polygon_points)
                    circle = plt.Circle((p1.x(), p1.y()), radius, color='#3cacae', fill=False, linestyle='--',
                                        linewidth=1)
                    plt.gca().add_artist(circle)
                if is_on_border(p2, polygon_points):
                    plt.scatter(p2.x(), p2.y(), color='#0e86d4', s=50, zorder=5)
                else:
                    radius = nearest_distance_to_border(p2, polygon_points)
                    circle = plt.Circle((p2.x(), p2.y()), radius, color='#3cacae', fill=False, linestyle='--',
                                        linewidth=1)
                    plt.gca().add_artist(circle)

        output_image = os.path.join(output_dir, f"split_{i + 1}_skeleton.png")
        plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved straight skeleton visualization for split {i + 1} to {output_image}")


def visualize_point_cloud_heatmap(point_cloud, output_image):
    points = point_cloud.points
    yz_coords = points[:, [0, 2]]

    center_yz = yz_coords.mean(axis=0)
    distances_from_center = np.linalg.norm(yz_coords - center_yz, axis=1)

    metric = distances_from_center

    metric_min, metric_max = metric.min(), metric.max()
    normalized_metric = (metric - metric_min) / (metric_max - metric_min)

    plotter = pv.Plotter(lighting="light_kit", off_screen=True)
    plotter.enable_depth_peeling()
    plotter.clear()
    cloud = pv.PolyData(points)
    cloud['Scalars'] = 1 - normalized_metric

    plotter.add_mesh(cloud, scalars='Scalars', cmap='coolwarm',
                     point_size=5,
                     render_points_as_spheres=True,
                     lighting=True)
    plane = pv.Plane(center=(0, 0, -0.1), i_size=10, j_size=10)
    plotter.add_mesh(plane, color='white', opacity=0.5)

    plotter.camera_position = [(-14.72563820212639, 1.2312697834334705, -16.003906806065096),
                               (-14.881514316376428, 1.944864587845622, -7.67068461272462),
                               (-0.00010408811382345371, 0.9963534069707393, -0.08532219865583247)]

    plotter.remove_legend()
    plotter.remove_scalar_bar()
    plotter.show(screenshot=output_image)
    print(f"Saved heatmap visualization to {output_image}")


def generate_gripper(center, rotation=None, width=2.0, height=0.2, depth=0.5):
    x, y, z = center

    half_width = width / 2
    finger_thickness = height / 2

    left_bounds = [
        x - half_width - height, x - half_width,
        y - finger_thickness, y + finger_thickness,
        z, z + depth
    ]

    right_bounds = [
        x + half_width, x + half_width + height,
        y - finger_thickness, y + finger_thickness,
        z, z + depth
    ]

    bottom_bounds = [
        x - half_width - finger_thickness * 2, x + half_width + finger_thickness * 2,
        y - finger_thickness, y + finger_thickness,
        z - height, z
    ]

    tail_bounds = [
        x - half_width / 6, x + half_width / 6,
        y - finger_thickness / 2, y + finger_thickness / 2,
        z - height - depth, z - height
    ]

    all_bounds = [left_bounds, right_bounds, bottom_bounds, tail_bounds]

    gripper_parts = [pv.Box(bounds=bounds) for bounds in all_bounds]

    if rotation is not None:
        if rotation.shape == (4,):
            rotation_matrix = Rotation.from_quat(rotation).as_matrix()
        elif rotation.shape == (3, 3):
            rotation_matrix = rotation
        else:
            raise ValueError("Rotation must be a 3x3 matrix or a quaternion (size 4).")

        for part in gripper_parts:
            part.points = (np.dot(part.points - center, rotation_matrix.T)) + center

    return gripper_parts


def point_cloud_grasp_grippers(main_cloud, output_dir, gripper_poses):
    os.makedirs(output_dir, exist_ok=True)
    plotter = pv.Plotter(lighting="light_kit", off_screen=True)
    plotter.enable_depth_peeling()

    plotter.add_mesh(main_cloud,
                     scalars='Colors',
                     rgb=True,
                     point_size=5,
                     render_points_as_spheres=True,
                     lighting=True,
                     ambient=0.4,
                     specular=0.3,
                     smooth_shading=True)

    for center, rotation in gripper_poses:
        gripper_parts = generate_gripper(center=center, rotation=rotation)
        for part in gripper_parts:
            plotter.add_mesh(part, color=[0.7, 0.7, 0.7])

    plotter.camera_position = [(-14.698993888987538, 1.1947323829760697, -17.75979311447645),
                               (-14.869161747592532, 2.3369579398410263, -7.704029494447265),
                               (0.0004116022225036794, 0.9936112145136801, -0.11285647954929352)]
    output_image = os.path.join(output_dir, "point_cloud_with_rotated_grippers.png")
    plotter.show(screenshot=output_image)
    print(f"Visualization saved to {output_image}")


if __name__ == "__main__":
    obj_file = "gs3d/Visualization/hourglass.obj"
    output_dir = "results/visualizations"
    primary_color = hex_to_rgb_array('#055c9d')
    split_colors = [
        hex_to_rgb_array('#003060'),
        hex_to_rgb_array('#0e86d4'),
        hex_to_rgb_array('#055c9d')
    ]

    main_cloud = sample_point_cloud_from_mesh(obj_file, primary_color, scale_factor=20.0)

    gripper_poses = [
        ((-14.7, 4, -8), Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()),
        ((-13.8, 2.3, -8), Rotation.from_euler('xyz', [0, 90, -180], degrees=True).as_quat()),
        ((-15.6, 2.3, -8), Rotation.from_euler('xyz', [0, -90, -180], degrees=True).as_quat())
    ]
    point_cloud_grasp_grippers(main_cloud, output_dir, gripper_poses)

    heatmap_output_image = os.path.join(output_dir, "point_cloud_heatmap.png")
    visualize_point_cloud_heatmap(main_cloud, heatmap_output_image)

    split_clouds = split_point_cloud(main_cloud, split_colors)

    apply_straight_skeleton_to_each_split(split_clouds, output_dir)

    combined_split_cloud = pv.PolyData()
    for sub_cloud in split_clouds:
        combined_split_cloud = combined_split_cloud.merge(sub_cloud)

    visualize_and_save_point_clouds([main_cloud, combined_split_cloud], output_dir, "hourglass")
