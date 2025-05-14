import math

import numpy as np
import skgeom as sg
import shapely.geometry
from sympy import cos, sin
from scipy.spatial import cKDTree


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def poly_iou(box1, box2):
    poly1 = shapely.geometry.Polygon(box1)
    poly2 = shapely.geometry.Polygon(box2)

    # Calculate the intersection area
    inter_area = poly1.intersection(poly2).area

    # Calculate the union area
    union_area = poly1.area + poly2.area - inter_area

    # Calculate the Intersection over Union
    iou = inter_area / union_area

    return iou


def calc_angle_for_bb(p1, p2):
    angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    if angle < 0:
        angle += 180
    return angle


def calculateDegree(ref_point, point1, point2):
    dx1, dy1 = point1[0] - ref_point[0], point1[1] - ref_point[1]
    dx2, dy2 = point2[0] - ref_point[0], point2[1] - ref_point[1]
    angle1 = math.degrees(math.atan2(dy1, dx1))
    angle2 = math.degrees(math.atan2(dy2, dx2))
    return abs(angle1 - angle2)


def four_courners_from_bb(p1, p2):
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    theta = calc_angle_for_bb(p1, p2)
    half_width = (distance(p1, p2) + 5) / 2
    half_height = 0.25 * (distance(p1, p2) + 5)
    corner1 = [x - half_width * cos(theta * math.pi / 180.0) - half_height * sin(theta * math.pi / 180.0),
               y - half_width * sin(theta * math.pi / 180.0) + half_height * cos(theta * math.pi / 180.0)]
    corner2 = [x + half_width * cos(theta * math.pi / 180.0) - half_height * sin(theta * math.pi / 180.0),
               y + half_width * sin(theta * math.pi / 180.0) + half_height * cos(theta * math.pi / 180.0)]
    corner3 = [x + half_width * cos(theta * math.pi / 180.0) + half_height * sin(theta * math.pi / 180.0),
               y + half_width * sin(theta * math.pi / 180.0) - half_height * cos(theta * math.pi / 180.0)]
    corner4 = [x - half_width * cos(theta * math.pi / 180.0) + half_height * sin(theta * math.pi / 180.0),
               y - half_width * sin(theta * math.pi / 180.0) - half_height * cos(theta * math.pi / 180.0)]

    return [corner1, corner2, corner3, corner4]


def del_list_indexes(l, id_to_del):
    somelist = [i for j, i in enumerate(l) if j not in id_to_del]
    return somelist


def calculate_angle(p1, p2):
    x_diff = (p2[0] - p1[0])
    y_diff = (p2[1] - p1[1])
    radians = math.atan2(y_diff, x_diff)
    angle = math.degrees(radians)
    if angle < 0:
        angle += 180
    return angle % 180


def groupContours(shape_poly, c_x, c_y, hole_flag):
    vertices = []
    for poly in shape_poly:
        x, y = poly.exterior.coords.xy
        for i in range(len(x)):
            vertices.append(((x[i], y[i]), (x[(i + 1) % len(x)], y[(i + 1) % len(x)])))
    groups = dict()
    for vertic in vertices:
        p1, p2 = vertic
        p1 = (float(p1[0]), float(p1[1]))
        p2 = (float(p2[0]), float(p2[1]))
        angle = calculate_angle(p1, p2)
        number_of_points = 50
        xs = np.linspace(p1[0], p2[0], number_of_points + 2)
        ys = np.linspace(p1[1], p2[1], number_of_points + 2)
        if angle not in groups.keys():
            groups[angle] = set()
        for i in range(len(xs)):
            groups[angle].add((xs[i], ys[i]))
    return groups


def find_closest_angle(groups, p1):
    total_dist = np.inf
    point_angle = 0

    p1 = np.array(p1)
    for angle, points in groups.items():
        tree = cKDTree(list(points))
        dist, _ = tree.query(p1)

        if dist < total_dist:
            total_dist = dist
            point_angle = angle

    return point_angle


def angle_dif(d1, d2):
    if abs(d1 - d2) > 90:
        return 180 - abs(d1 - d2)
    else:
        return abs(d1 - d2)


def simplify_polygon2(polygon, max_iterations=25, max_points=100):
    if polygon.is_simple() and len(polygon) < max_points:
        return polygon

    new_poly = polygon
    iterations = 0

    while (not new_poly.is_simple() or not len(new_poly) < max_points) and iterations < max_iterations and len(
            new_poly) > 2:
        new_poly = sg.simplify(new_poly, 0.1, "ratio")
        iterations += 1

    return new_poly


def simplify_polygon(polygon):
    if polygon.is_simple():
        return polygon
    points = [(float(point.x()), float(point.y())) for point in polygon.vertices]

    new_poly = polygon
    while not new_poly.is_simple():
        dists = []
        for i in range(len(points)):
            dists.append((points[i][0] - points[(i + 1) % len(points)][0]) ** 2 + (
                        points[i][1] - points[(i + 1) % len(points)][1]) ** 2)
        dists = np.array(dists)
        m = np.argmin(dists)
        del points[m]
        new_poly = sg.Polygon(points)

    return new_poly


def generatePointOnLine(point1, point2, dist):
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        if y1 < y2:
            return (x1, y1 - dist), (x2, y2 + dist)
        else:
            return (x1, y1 + dist), (x2, y2 - dist)
    dx1 = -1 if x2 - x1 > 0 else 1
    dy1 = -1 if y2 - y1 > 0 else 1

    angle = calc_angle_for_bb(point1, point2)
    dx = abs(dist * math.cos((angle * np.pi) / 180))
    dy = abs(dist * math.sin((angle * np.pi) / 180))

    new_x1 = x1 + (dx * dx1)
    new_y1 = y1 + (dy * dy1)
    new_x2 = x2 + (-1 * dx * dx1)
    new_y2 = y2 + (-1 * dy * dy1)
    return (new_x1, new_y1), (new_x2, new_y2)


def isHoleValid(mainPoly, holePolygon):
    if (holePolygon.intersects(mainPoly)) and not mainPoly.contains(holePolygon):
        return False
    return True
