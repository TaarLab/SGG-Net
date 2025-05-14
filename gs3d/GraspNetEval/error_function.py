import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance

@dataclass
class GripperParameters:
    max_gripper_opening: float
    min_gripper_opening: float
    current_gripper_z_angle: float

def calculate_score(point1, point2) -> float:
    return calculate_exploitation_error(point1, point2, -float("inf"), GripperParameters(0.1, 0.005, 0)) - MAX_EXPLOITATION_ERROR

def calculate_exploitation_error(point1, point2, acceptable_error: float, parameters: GripperParameters) -> float:
    error = 0.0
    for error_function in exploitation_error_functions:
        e = error_function(point1, point2, parameters)
        if e < acceptable_error or error + e < acceptable_error:
            error = MAX_EXPLOITATION_ERROR
            break
        else:
            error += e
    return error


def normal_error(point1, point2, parameters: GripperParameters):
    position1 = np.array(point1[:3])
    position2 = np.array(point2[:3])
    normal1 = np.array(point1[3:])
    normal2 = np.array(point2[3:])

    vector_p1_to_p2 = position2 - position1
    normalized_vector = normalize_vector(vector_p1_to_p2)

    # Make sure the normal are not point to each other
    if np.linalg.norm(normalized_vector + normal1) > 1:
        return -1

    dot_product = np.dot(normal1, normal2)
    return -normalize(dot_product + 1, 1)


def gripper_opening_error(point1, point2, parameters: GripperParameters):
    position1 = np.array(point1[:3])
    position2 = np.array(point2[:3])
    dist = distance.euclidean(position1, position2)

    # If the distance is more than the maximum gripper opening or less than the minimum gripper opening
    # we apply an exponential function to give a continuous negative error.
    if dist > parameters.max_gripper_opening or dist < parameters.min_gripper_opening * 6:
        error = -1
    else:
        error = 0

    return error


def gripper_opening_exploration_error(point1, point2, parameters: GripperParameters):
    position1 = np.array(point1[:3])
    position2 = np.array(point2[:3])
    dist = distance.euclidean(position1, position2)

    if dist > parameters.max_gripper_opening:
        error = 0
    else:
        error = -normalize(parameters.max_gripper_opening - dist, parameters.max_gripper_opening)

    return error


# Check the two point grasp and the average normal of two point are parallel or no
def grasp_angle_error(point1, point2, parameters: GripperParameters):
    position1 = np.array(point1[:3])
    position2 = np.array(point2[:3])

    normal1 = np.array(point1[3:])
    normal2 = np.array(point2[3:])

    dot_product = np.dot(normal1, normal2)

    normal1 *= 1 if dot_product >= 0 else -1

    normal = normalize_vector((normal1 + normal2) / 2)

    # A0, B0, C0 = normal
    # x0, y0, z0 = position1
    # D0 = A0 * x0 + B0 * y0 + C0 * z0
    # plane_1 = f"{A0}x + {B0}y + {C0}z = {D0}"

    vector_p1_to_p2 = position2 - position1
    normalized_vector = normalize_vector(vector_p1_to_p2)

    # A0, B0, C0 = normalized_vector
    # x0, y0, z0 = position2
    # D0 = A0 * x0 + B0 * y0 + C0 * z0
    # plane_2 = f"{A0}x + {B0}y + {C0}z = {D0}"

    dot_product = abs(np.dot(normalized_vector, normal))

    return -normalize(1 - dot_product, 0.5)


def gripper_angle_error(point1, point2, parameters: GripperParameters):
    position1 = np.array(point1[:3])
    position2 = np.array(point2[:3])

    vector_p1_to_p2 = position2 - position1
    normalized_vector = normalize_vector(vector_p1_to_p2)

    target_vector = normalize_vector(np.array([1, 0, normalized_vector[2]]))
    gripper_vector = normalize_vector(np.array([1, 0, np.sin(parameters.current_gripper_z_angle)]))

    dot_product = abs(np.dot(target_vector, gripper_vector))

    return -normalize(1 - dot_product, 1)


# The min error for normal distance is -5, and it's relu function that avoid positive error
def plane_distance(plane_position, plane_normal, point_position):
    x0, y0, z0 = plane_position
    x1, y1, z1 = point_position
    A0, B0, C0 = plane_normal
    D0 = A0 * x0 + B0 * y0 + C0 * z0

    # plane_1 = f"{A0}x + {B0}y + {C0}z = {D0}"
    # point_1 = f"Point({{ {x0}, {y0}, {z0} }})"
    # point_2 = f"Point({{ {x1}, {y1}, {z1} }})"

    return abs(A0 * x1 + B0 * y1 + C0 * z1 - D0) / math.sqrt(A0 ** 2 + B0 ** 2 + C0 ** 2)


def plane_distance_error(point1, point2, parameters: GripperParameters):
    position1 = np.array(point1[:3])
    position2 = np.array(point2[:3])

    normal1 = np.array(point1[3:])
    normal2 = np.array(point2[3:])

    plane_distance_1 = plane_distance(position1, normal1, position2)
    plane_distance_2 = plane_distance(position2, normal2, position1)
    pd = min(plane_distance_1, plane_distance_2)
    if pd < parameters.min_gripper_opening * 2:
        return -normalize(parameters.min_gripper_opening * 2 - pd, parameters.min_gripper_opening)
    else:
        return 0


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize(value, k):
    return min(max(0, value / k) - 1, 0) + 1


exploitation_error_functions = [
    gripper_opening_error,
    normal_error,
    plane_distance_error,
    grasp_angle_error,
    # gripper_angle_error
]
MAX_EXPLOITATION_ERROR = -len(exploitation_error_functions) # based on count of errors formula