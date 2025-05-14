
import open3d as o3d
import numpy as np
import zarr
from scipy.spatial.transform import Rotation


def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    # x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    
    # height=0.001
    # finger_width = 0.001
    # tail_length = 0.04
    # depth_base = abs(0.06-depth)
    
    # if color is not None:
    #     color_r, color_g, color_b = color
    # else:
    color_r = score # red for high score
    color_g = 0
    color_b = 1 - score # blue for low score
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + np.array(center)
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for i in range(len(vertices))])
    
    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def plot_grasps(combined_cloud , graspPoses , max_grasps=10):
    np.random.shuffle(graspPoses)
    grippers=[]
    num_grp = min(len(graspPoses ), max_grasps)
    for i in range(num_grp):
        g0 = graspPoses[i]
        grp = plot_gripper_pro_max(g0['translation'], g0['rotation'], g0['width'], g0['depth'])
        grippers.append(grp)
    o3d.visualization.draw_geometries([combined_cloud ]+ grippers)


# def findGraspCircles(self, polygon, skeleton, allContourPoints):  
#     centersX, centersY, vt = [], [], []
#     centersXMiddle,centersYMiddle = [],[]
#     vertices = []
#     circlePolies = []
#     for h in skeleton.halfedges:
#         if h.is_bisector:
#             p1, p2 = h.vertex.point, h.opposite.vertex.point
#             x1 = float(p1.x())
#             x2 = float(p2.x())
#             y1 = float(p1.y())
#             y2 = float(p2.y())

def cross_product(a, b):
    return np.cross(a, b)


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


     
if __name__ == "__main__":
    pcd_np = np.load("visualization/mmd.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    rotation = vector_to_quaternion([1,0,0], np.array([0,0,0]) - np.array([0.1,0,0]))
    plot_grasps(pcd, [{'translation':[0,0,0],'rotation':Rotation.from_quat(rotation).as_matrix(),'width':0.1,'depth':0}],1)
