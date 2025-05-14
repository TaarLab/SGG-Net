import open3d as o3d
import numpy as np

from gs3d.utils.utils import get_points_on_line
from gs3d.utils.keypoint_utils import distance








def get_mid_points(skeleton):
    skeleton_points = []
    for h in skeleton.halfedges:
        if h.is_bisector:
            p1, p2 = h.vertex.point, h.opposite.vertex.point
            x1 = float(p1.x())
            x2 = float(p2.x())
            y1 = float(p1.y())
            y2 = float(p2.y())
            points = get_points_on_line(x1,y1,x2,y2,max(2, int(distance((x1,y1),(x2,y2)) / 0.001)))
            skeleton_points += points
    return skeleton_points


def return_skeleton_points_2d(skeleton, height):
    mid_points = get_mid_points(skeleton)
    return [(p[0],p[1],height) for p in mid_points]




def plot_3d_skeleton(pcd_points,skeleton_3d):
    # Create a point cloud from the numpy array 
        
    pcd_o3d = o3d.geometry.PointCloud()  
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_points)
    pcd_o3d.paint_uniform_color([1, 0.5, 0])  # Set the point color to black  

    
    
    # Create a visualization object  
    vis = o3d.visualization.Visualizer()  
    vis.create_window()  

    # Add the point cloud  
    vis.add_geometry(pcd_o3d)  

    # Add the skeleton points  
    skeleton_pcd = o3d.geometry.PointCloud()  
    skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_3d)  
    skeleton_pcd.paint_uniform_color([0, 0, 0])  # Set the point color to black  
    vis.add_geometry(skeleton_pcd)  

    # Render the scene  
    vis.run()  
    vis.destroy_window() 
    

def plot_3d_section(pcd_points,z_min,z_max):
    # Extract the section points  
    section_points = pcd_points[np.logical_and(pcd_points[:, 2] > z_min, pcd_points[:, 2] < z_max)]  
    pcd_points = pcd_points[np.logical_or(pcd_points[:,2] <= z_min,pcd_points[:,2] >= z_max)]
    # Create the point cloud  
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(pcd_points)  
    pcd.paint_uniform_color([1, 0.5, 0])  # Set the point color to black  


    # Create the section point cloud  
    section_pcd = o3d.geometry.PointCloud()  
    section_pcd.points = o3d.utility.Vector3dVector(section_points)  
    section_pcd.paint_uniform_color([1,0.01,0.01])
    box = o3d.geometry.TriangleMesh.create_box(width=0.08   , height=0.08, depth=0.0005)  

    box.paint_uniform_color([0.5, 0.5, 0.5])  # Set the color to gray 
    # box.transluca
    current_position = box.get_center()  
 

    # Define the target position (x, y, z)  
    target_position = [-0.006,0.006 , z_min]  

    # Compute the translation vector  
    translation_vector = [target_position[0] - current_position[0],  
                    target_position[1] - current_position[1],  
                    target_position[2] - current_position[2]]  
    box.translate(translation_vector)



    # Visualize the point cloud, section, and triangle  
    o3d.visualization.draw_geometries([pcd, section_pcd, box])  