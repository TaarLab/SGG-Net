import math

import cv2
import numpy as np
import skgeom as sg
import shapely.geometry
from skgeom.draw import draw
import matplotlib.pyplot as plt
from alphashape import alphashape
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from gs3d.GraspNetEval.error_function import calculate_score
from gs3d.utils.keypoint_utils import generatePointOnLine, angle_dif, calculate_angle, calculateDegree, \
    del_list_indexes, distance, isHoleValid, four_courners_from_bb, groupContours, find_closest_angle, \
    simplify_polygon2, poly_iou, simplify_polygon

DIST_THRESH = 30


def polygon_to_vertices(polygon):
    return np.array([(v.x(), v.y()) for v in polygon.vertices], dtype=np.float32).tolist()


def vertices_to_polygon(vertices):
    return sg.Polygon([sg.Point2(x, y) for x, y in vertices])


def plot_contour_points_with_angles(contour_points, contourGroups):
    # Extract the x and y coordinates from the contour_points array  
    x_coords = np.array(contour_points)[:, 0].astype('float64')
    y_coords = np.array(contour_points)[:, 1].astype('float64')

    # Create the plot  
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_coords, y_coords, color='blue')

    # Add the angle labels  
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        angle = contourGroups[i]
        ax.text(x, y, f"{angle:.2f}°", ha='center', va='bottom', fontsize=8)

        # Set the plot title and axis labels
    ax.set_title("Contour Points with Angles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Show the plot  
    plt.show()


class KeypointGeneration:
    def __init__(self, debugMode=False, considerGripperValidity=False, pair_check_extension=0.001, pair_itrs=5):
        self.considerGripperValidity = considerGripperValidity
        self.pair_check_extension = pair_check_extension
        self.debugMode = debugMode
        self.angleWithCenter = 75
        self.pair_itrs = pair_itrs
        self.orthogonalAngleThresh = 15
        self.parallelAngleThresh = 20

    def makeGradiantSkeleton(self, poly):
        OFFSET_MULTIPLIER = 0.1

        # Assuming 'poly' is the
        straight_skeleton = sg.skeleton.create_interior_straight_skeleton(poly)
        if straight_skeleton is None:
            return None
        #  original polygon

        # Set the minimum and maximum offset values

        done = False
        offset = 1
        array = np.zeros((256, 256)) - 1
        while not done:
            poly = straight_skeleton.offset_polygons(offset * OFFSET_MULTIPLIER)
            if (len(poly) == 0):
                done = True
            for poly_in in poly:
                points = list(poly_in.vertices)
                x = [float(p.x()) for p in points]
                y = [float(p.y()) for p in points]
                x_s = np.array([float(x_coord) for x_coord in x])
                y_s = np.array([float(y_coord) for y_coord in y])
                points = [[x_s[i], y_s[i]] for i in range(len(x_s))]
                mask = np.zeros([256, 256])
                mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=1)
                array[mask.astype(bool)] = (offset) * 0.1
            offset += 1
            if (offset > 100):
                break
        normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
        return normalized_array

    def findMinDistanceToContourPoints(self, contour_points, centers):
        main_radius = np.inf
        for point in contour_points:
            point = np.reshape(point, (1, 2))
            my_points = point[0][0], point[0][1]
            my_center = float(centers[0]), float(centers[1])
            dist = distance(my_points, my_center)
            if dist < main_radius:
                main_radius = dist
        return main_radius

    def does_intersect_gripper(self, gripper, polies):
        for poly in polies:
            if (gripper.intersects(poly)):
                return True
        return False

    def check_for_empty_gripper_space(self, p1, p2, polies):
        new_p1, new_p2 = generatePointOnLine(p1, p2, 7)
        if (len(polies) > 1):
            return True, p1, p2

        for i in range(5):
            gripper = shapely.geometry.Point(new_p1[0], new_p1[1]).buffer(5)
            if (self.does_intersect_gripper(gripper, polies)):
                return False, new_p1, new_p2
            else:
                new_p1, _ = generatePointOnLine(new_p1, new_p2, 2)

        for i in range(5):
            gripper = shapely.geometry.Point(new_p2[0], new_p2[1]).buffer(5)
            if (self.does_intersect_gripper(gripper, polies)):
                return False, new_p1, new_p2
            else:
                _, new_p2 = generatePointOnLine(new_p1, new_p2, 2)

        return True, p1, p2

    def check_validity(self, p1, p2, c_x, c_y, contour_groups):

        dists = np.power(p1[0] - c_x, 2) + np.power(p1[1] - c_y, 2)
        if dists.shape[0] < 4:
            return False
        min_indexes = np.argpartition(dists, 4)[:4]
        x_s1 = c_x[min_indexes]
        y_s1 = c_y[min_indexes]
        dists = np.power(p2[0] - c_x, 2) + np.power(p2[1] - c_y, 2)
        min_indexes = np.argpartition(dists, 4)[:4]
        x_s2 = c_x[min_indexes]
        y_s2 = c_y[min_indexes]
        p1_angles = []
        p2_angles = []
        i = 0
        for angle in contour_groups.keys():
            if (x_s1[i], y_s1[i]) in contour_groups[angle]:
                p1_angles.append(angle)
            if (x_s2[i], y_s2[i]) in contour_groups[angle]:
                p2_angles.append(angle)
        p1_angles.sort()
        p2_angles.sort()
        if p1_angles == [] or p2_angles == []:
            return False
        if angle_dif(p1_angles[0], p1_angles[-1]) < self.parallelAngleThresh and angle_dif(p2_angles[0], p2_angles[
            -1]) < self.parallelAngleThresh:
            deg = calculate_angle(p1, p2)
            if 90 - self.orthogonalAngleThresh <= angle_dif(deg, sum(p1_angles) / len(
                    p1_angles)) <= 90 + self.orthogonalAngleThresh and 90 - self.orthogonalAngleThresh <= angle_dif(deg,
                                                                                                                    sum(p2_angles) / len(
                                                                                                                        p2_angles)) <= 90 + self.orthogonalAngleThresh:
                return True
        return False

    def process(self, points, center, contoursX, contoursY, contour_groups, badPoints, polies):
        # chosen_points = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if (points[i], points[j]) in badPoints:
                    continue
                degree = calculateDegree(center, points[i], points[j])
                if 180 - self.angleWithCenter <= degree <= 180 + self.angleWithCenter:
                    if self.check_validity(points[i], points[j], contoursX, contoursY, contour_groups):
                        if self.considerGripperValidity:
                            gripper_valid, p1, p2 = self.check_for_empty_gripper_space(points[i], points[j], polies)
                            if gripper_valid:
                                return True, p1, p2
                            else:
                                badPoints.add((points[i], points[j]))
                                badPoints.add((points[j], points[i]))
                        else:
                            return True, p1, p2

                    else:
                        badPoints.add((points[i], points[j]))
                        badPoints.add((points[j], points[i]))
                else:
                    badPoints.add((points[i], points[j]))
                    badPoints.add((points[j], points[i]))

        return False, 0, 0

    def find_closest_point(self, inside_points, grasp_point1, normals, radius=0.02):
        distances_2d = np.linalg.norm(inside_points[:, :2] - grasp_point1[:2], axis=1)

        close_2d_points_indices = np.where(distances_2d <= radius)[0]

        if len(close_2d_points_indices) > 0:
            zs = inside_points[close_2d_points_indices, 2]
            estimated_z = (zs.max() + zs.min()) / 2

            closest_index_2d = close_2d_points_indices[np.argmin(distances_2d[close_2d_points_indices])]
            closest_point = np.copy(inside_points[closest_index_2d])
            closest_point[2] = estimated_z
        else:
            closest_index_2d = np.argmin(distances_2d)
            closest_point = inside_points[closest_index_2d]

        closest_normal = normals[closest_index_2d]

        return closest_point, closest_normal

    def find_best_scored_pair(self, pairs, normals):
        top_pairs = []
        top_scores = []

        for i, pair in enumerate(pairs):
            c1 = np.concatenate((np.array(pair[0]), np.array(normals[i][0])))
            c2 = np.concatenate((np.array(pair[1]), np.array(normals[i][1])))
            score = calculate_score(c1, c2)
            top_scores.append(score)
            top_pairs.append(pair)

        return top_pairs, top_scores

    def check_scored_validity(self, p1, p2, contoursX, contoursY, contourGroups, ind1, ind2):
        angle1 = contourGroups[ind1]
        angle2 = contourGroups[ind2]
        deg = calculate_angle(p1, p2)
        if 90 - self.orthogonalAngleThresh <= angle_dif(deg,
                                                        angle1) <= 90 + self.orthogonalAngleThresh and 90 - self.orthogonalAngleThresh <= angle_dif(
            deg, angle2) <= 90 + self.orthogonalAngleThresh:
            return True
        else:
            return False

    def keypoint_scored(self, contour_points, centers, radiuses, cropped_pcd, height, contourGroups):
        centersX = centers[0].astype('float64')
        centersY = centers[1].astype('float64')
        contoursX = np.array(contour_points)[:, 0].astype('float64')
        contoursY = np.array(contour_points)[:, 1].astype('float64')
        results = []
        results_score = []
        margin = 0.005
        for i in range(len(centersX)):
            chosen_points = []
            center = (centersX[i], centersY[i])
            if radiuses[i] == 0:
                continue
            dists = np.sqrt(np.power(centersX[i] - contoursX, 2) + np.power(centersY[i] - contoursY, 2))
            indices = np.where(dists <= radiuses[i] + margin)[0]
            x = contoursX[indices]
            y = contoursY[indices]
            points = list(set(zip(x, y)))

            for k in range(len(points) - 1):
                for j in range(k + 1, len(points)):
                    degree = calculateDegree(center, points[k], points[j])
                    if 180 - self.angleWithCenter <= degree <= 180 + self.angleWithCenter:
                        if self.check_scored_validity(points[k], points[j], contoursX, contoursY, contourGroups,
                                                      indices[k], indices[j]):
                            chosen_points.append((points[k], points[j]))
            d3_pairs = []
            d3_normals = []
            for pair in chosen_points:
                p1, p2 = pair
                p1 = [p1[0], p1[1], height]
                p2 = [p2[0], p2[1], height]
                grasp_point1, normal_point1 = self.find_closest_point(np.array(cropped_pcd.points), p1,
                                                                      np.array(cropped_pcd.normals))
                grasp_point2, normal_point2 = self.find_closest_point(np.array(cropped_pcd.points), p2,
                                                                      np.array(cropped_pcd.normals))
                d3_pairs.append((grasp_point1, grasp_point2))
                d3_normals.append((normal_point1, normal_point2))
                if self.debugMode:
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

            if len(d3_normals) > 0:
                top_points, top_scores = self.find_best_scored_pair(d3_pairs, d3_normals)
                results += top_points
                results_score += top_scores
        return results, results_score

    def keypoint(self, contour_points, centers, radiuses, contour_groups, polies):
        centersX = centers[0].astype('float64')
        centersY = centers[1].astype('float64')
        contoursX = np.array(contour_points)[:, 0].astype('float64')
        contoursY = np.array(contour_points)[:, 1].astype('float64')
        result = []
        for i in range(len(centersX)):
            bad_points = set()
            if radiuses[i] == 0:
                continue
            margin, valid, iterations = 0.0, 0, 0
            while iterations <= self.pair_itrs:
                dists = np.sqrt(np.power(centersX[i] - contoursX, 2) + np.power(centersY[i] - contoursY, 2))
                x = contoursX[np.where(dists <= radiuses[i] + margin)[0]]
                y = contoursY[np.where(dists <= radiuses[i] + margin)[0]]
                points = list(zip(x, y))
                valid, point1, point2 = self.process(points, (centersX[i], centersY[i]), contoursX, contoursY,
                                                     contour_groups, bad_points, polies)
                if valid:
                    result.append((point1, point2))
                    break
                else:
                    margin += self.pair_check_extension

                iterations += 1

        return result

    def checkForRedundantCircles(self, circlePolies, x, y, r):
        removable_indices = []
        new_poly = shapely.Point(1, 1)
        for i, poly in enumerate(circlePolies):
            if poly.contains(new_poly):
                return [], False
            if new_poly.contains(poly):
                removable_indices.append(i)
        return removable_indices, True

    def findGraspCircles(self, polygon, skeleton, allContourPoints):
        if self.debugMode:
            draw(polygon)

        centersX, centersY, vt = [], [], []
        centersXMiddle, centersYMiddle = [], []
        vertices = []
        circlePolies = []
        for v in skeleton.vertices:
            if float(v.time) == 0:
                continue
            # if (check_iou_in_list(centersX,centersY,vt,float(v.point.x()),float(v.point.y()),float(v.time))):
            x = float(v.point.x())
            y = float(v.point.y())
            r = float(v.time)
            deleteIndices, validCenter = self.checkForRedundantCircles(circlePolies, x, y, r)
            if validCenter:
                centersX = del_list_indexes(centersX, deleteIndices)
                centersY = del_list_indexes(centersY, deleteIndices)
                vt = del_list_indexes(vt, deleteIndices)
                circlePolies = del_list_indexes(circlePolies, deleteIndices)

                centersX.append(float(v.point.x()))
                centersY.append(float(v.point.y()))
                vt.append(float(v.time))
                circlePolies.append(shapely.Point(x, y).buffer(r))
                vertices.append((float(v.point.x()), float(v.point.y())))
                # if self.debugMode:
                #     plt.gcf().gca().add_artist(plt.Circle((v.point.x(), v.point.y()), v.time, color='red', fill=False))
            vertices.append((float(v.point.x()), float(v.point.y())))

        for h in skeleton.halfedges:
            if h.is_bisector:
                p1, p2 = h.vertex.point, h.opposite.vertex.point
                x1 = float(p1.x())
                x2 = float(p2.x())
                y1 = float(p1.y())
                y2 = float(p2.y())
                if (distance((x1, y1), (x2, y2)) >= DIST_THRESH and float(p1.x()), float(p1.y())) in vertices and (
                        float(p2.x()), float(p2.y())) in vertices:
                    num_points = int(distance((x1, y1), (x2, y2)) // (DIST_THRESH / 2)) + 1
                    for i in range(num_points):
                        t = i / (num_points - 1)
                        xMid = (1 - t) * x1 + t * x2
                        yMid = (1 - t) * y1 + t * y2
                        centersXMiddle.append(xMid)
                        centersYMiddle.append(yMid)
                # if self.debugMode:
                #     plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'y-', lw=2)

        middles = np.array([centersXMiddle, centersYMiddle])
        if self.debugMode:

            for i in range(middles.shape[1]):
                radius = self.findMinDistanceToContourPoints(allContourPoints, [middles[0][i], middles[1][i]])
                plt.gcf().gca().add_artist(plt.Circle((middles[0][i], middles[1][i]), radius, color='red', fill=False))

            plt.scatter(centersXMiddle, centersYMiddle, color="black", zorder=2)

        return centersX, centersY, vt, middles

    def findContourSize(self, x):
        x_size = [np.shape(x)[i] for i in range(3) if np.shape(x)[i] > 1]
        return x_size

    def createContours(self, path):
        img1 = cv2.imread(path)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(img1_gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        saved_approx, whole_points = [], []

        valid_contours = []
        for contour in contours:
            if np.shape(contour)[0] > 20:
                whole_points.append(contour)
                epsilon = 0.008 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                saved_approx.append(approx)
                valid_contours.append(contour)
        return valid_contours, saved_approx

    def createPolygons(self, path):
        contours, saved_approx = self.createContours(path)
        if len(contours) == 0:
            return None

        contoursNumber = np.shape(contours)[0]
        mainPolySize = self.findContourSize(saved_approx[0])
        mainPolyPoints = np.reshape(saved_approx[0], (mainPolySize[0], mainPolySize[1]))[::-1]
        mainPolygon = shapely.Polygon(mainPolyPoints)
        mainContourShape = self.findContourSize(contours[0])
        mainContourPoints = np.reshape(contours[0], (mainContourShape[0], mainContourShape[1]))
        allPolygons = []
        holeSGPolygons = []
        allContourPoints = mainContourPoints
        hasValidHole = False

        allPolygons.append(mainPolygon)

        if contoursNumber > 1:
            for i in range(contoursNumber - 1):
                holePolyVeritcsNumber = self.findContourSize(saved_approx[i + 1])
                if holePolyVeritcsNumber[0] >= 3:
                    holePolygonPoints = np.reshape(saved_approx[i + 1],
                                                   (holePolyVeritcsNumber[0], holePolyVeritcsNumber[1]))[::-1]
                    holePolygon = shapely.Polygon(holePolygonPoints)
                    if not isHoleValid(mainPolygon, holePolygon):
                        continue

                    hasValidHole = True
                    allPolygons.append(holePolygon)
                    holePolygon = sg.Polygon(holePolygonPoints)
                    holeSGPolygons.append(holePolygon)
                    holeContourShape = self.findContourSize(contours[i + 1])
                    holeContourPointsInversed = np.reshape(contours[i + 1], (holeContourShape[0], holeContourShape[1]))[
                                                ::-1]
                    allContourPoints = np.concatenate((allContourPoints, holeContourPointsInversed), axis=0)

        if contoursNumber <= 1 or not hasValidHole:
            finalPolygon = sg.Polygon(mainPolyPoints)
            finalPolygon = simplify_polygon(finalPolygon)
        else:
            main_poly = sg.Polygon(mainPolyPoints)
            finalPolygon = sg.PolygonWithHoles(main_poly, holeSGPolygons)

        return allContourPoints, allPolygons, finalPolygon

    def generateGraspCirclesFromSkeleton(self, finalPolygon, straightSkeleton, allContourPoints):

        centersX, centersY, radiuses, middlesPoints = self.findGraspCircles(finalPolygon, straightSkeleton,
                                                                            allContourPoints)

        midllePointRadiuses = []

        for i in range(middlesPoints.shape[1]):
            radius = self.findMinDistanceToContourPoints(allContourPoints, [middlesPoints[0][i], middlesPoints[1][i]])
            midllePointRadiuses.append(radius)

        finalCenters = np.array([centersX, centersY])
        return finalCenters, radiuses, middlesPoints, midllePointRadiuses

    def filterGraspPosesWithIOU(self, results1, results2):
        results2 = list(set(results2))
        results = results1.copy()

        res1_bb = []
        res2_bb = []

        for res in results2:
            p1, p2 = res
            corners = four_courners_from_bb(p1, p2)
            res2_bb.append(corners)

        for res in results1:
            p1, p2 = res
            corners = four_courners_from_bb(p1, p2)
            res1_bb.append(corners)

        for i, corner2 in enumerate(res2_bb):
            ok = 1
            for corner1 in res1_bb:
                if poly_iou(corner2, corner1) > 0.0:
                    ok = 0
                    break
            if ok:
                results.append(results2[i])

        return results

    def makeContourAngleGroup(self, allContourPoints, allPolygons):
        hasHole = True if len(allPolygons) > 1 else False
        contourPointsX, contourPointsY = np.array(allContourPoints)[:, 0], np.array(allContourPoints)[:, 1]
        groups = groupContours(allPolygons, contourPointsX, contourPointsY, hole_flag=hasHole)
        contourGroups = dict()
        for i in range(len(contourPointsX)):
            angle = find_closest_angle(groups, (contourPointsX[i], contourPointsY[i]))
            if angle in contourGroups:
                contourGroups[angle].add((contourPointsX[i], contourPointsY[i]))
            else:
                contourGroups[angle] = {(contourPointsX[i], contourPointsY[i])}
        return contourGroups

    def returnGraspPoses(self, path):
        allContourPoints, allPolygons, finalPolygon = self.createPolygons(path)

        straightSkeleton = sg.skeleton.create_interior_straight_skeleton(finalPolygon)
        gradient = self.makeGradiantSkeleton(finalPolygon)
        if gradient is None:
            return [], None
        # max_indices = np.where(gradient == np.max(gradient))
        # y_coords, x_coords = max_indices
        # plt.scatter(x_coords,y_coords,c="black",s =0.1)
        # plt.imshow(gradient)
        # plt.show()
        # for offset_poly in straightSkeleton.offset_polygons(5):
        # draw(offset_poly, facecolor="red")
        # plt.show()

        finalGraspPoses = []
        try:
            finalCenters, radiuses, middlesPoints, midllePointRadiuses = self.generateGraspCirclesFromSkeleton(
                finalPolygon, straightSkeleton, allContourPoints)

            contourGroups = self.makeContourAngleGroup(allContourPoints, allPolygons)
            mainGraspPoses = self.keypoint(allContourPoints, finalCenters, radiuses, contourGroups, allPolygons)
            middleGraspPoses = self.keypoint(allContourPoints, middlesPoints, midllePointRadiuses, contourGroups,
                                             allPolygons)
            finalGraspPoses = self.filterGraspPosesWithIOU(mainGraspPoses, middleGraspPoses)
        except:
            print("invalid Straight Skeleton")
            # SG library sometimes generates invalid skeletons that cause error in our functions
        return finalGraspPoses, gradient

    def get_points_on_line(self, x1, y1, x2, y2, n):
        points = []
        for i in range(n):
            t = i / (n - 1)
            x = (1 - t) * x1 + t * x2
            y = (1 - t) * y1 + t * y2
            if (x == x1 and y == y1) or (x == x2 and y == y2):
                continue
            points.append((x, y))
        return points

    def generate_contour_points_from_polygon(self, poly_hole):
        coords = poly_hole.coords
        vertices_number = coords.shape[0]
        contour_points = np.array([])
        contourGroups = dict()
        for i in range(vertices_number):
            p1 = coords[i, :]
            p2 = coords[(i + 1) % vertices_number, :]
            angle = calculate_angle(p1, p2)
            dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
            points = self.get_points_on_line(p1[0], p1[1], p2[0], p2[1], max(3, int(dist / 0.01)))
            points_array = np.array(points)
            if points_array.shape[0] > 0:
                contourGroups[angle] = set(
                    [(points_array[i, 0], points_array[i, 1]) for i in range(points_array.shape[0])])
            if contour_points.size == 0:
                contour_points = points_array
            else:
                contour_points = np.concatenate((contour_points, points_array), axis=0)
        return contour_points, contourGroups

    def contour_point_generation_scored(self, poly_hole):
        coords = poly_hole.coords
        vertices_number = coords.shape[0]
        contour_points = np.array([])
        contourGroups = []
        for i in range(vertices_number):
            p1 = coords[i, :]
            p2 = coords[(i + 1) % vertices_number, :]
            angle = calculate_angle(p1, p2)
            dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
            points = self.get_points_on_line(p1[0], p1[1], p2[0], p2[1], max(4, int(dist / 0.01)))
            points_array = np.array(points)
            contourGroups += [angle for i in range(points_array.shape[0])]
            if points_array.shape[0] == 0:
                continue
            if contour_points.size == 0:
                contour_points = points_array
            else:
                contour_points = np.concatenate((contour_points, points_array), axis=0)
        return contour_points, contourGroups

    def split_point_cloud_by_height(self, points, step_height=0.02):
        heights = points[:, 2]
        min_height = np.min(heights)
        max_height = np.max(heights)

        selected_points = []
        cross_heights = []

        current_height = min_height
        while current_height <= max_height:
            upper_height = current_height + step_height
            section_points = points[(heights > current_height) & (heights <= upper_height)]
            if len(section_points) == 0:
                current_height = upper_height
                continue
            # if section_points.shape[0] > 100:
            selected_points.append(section_points)
            # cross_heights.append((current_height + upper_height) / 2)
            cross_heights.append(((section_points[:, 2].max() + section_points[:, 2].min()) / 2))
            # cross_heights.append(upper_height)

            current_height = upper_height

        return selected_points, cross_heights

    def down_sample_circles(self, final_centers, radiuses):

        # Sort the circles by their radius in ascending order
        indices = np.argsort(radiuses)
        final_centers = final_centers[:, indices]
        radiuses = [radiuses[i] for i in indices]

        # Initialize the downsampled lists
        downsampled_centers = []
        downsampled_radiuses = []

        # Iterate through the sorted circles
        for i in range(final_centers.shape[1]):
            # Check if the current circle is close to any of the previously added circles
            is_close = False
            for j in range(len(downsampled_centers)):
                if np.linalg.norm(final_centers[:, i] - downsampled_centers[j]) + 0.01 <= downsampled_radiuses[j] + \
                        radiuses[i]:
                    is_close = True
                    break

                    # If the current circle is not close to any previous circles, add it to the downsampled lists
            if not is_close:
                downsampled_centers.append(final_centers[:, i])
                downsampled_radiuses.append(radiuses[i])

        return np.array(downsampled_centers).T, downsampled_radiuses

    def create_skeleton(self, polygon):
        polygon = simplify_polygon2(polygon)
        vertices = polygon_to_vertices(polygon)
        polygon = vertices_to_polygon(vertices)
        polygon = simplify_polygon2(polygon)
        if not polygon.is_simple():
            return None, polygon
        return sg.skeleton.create_interior_straight_skeleton(polygon), polygon

    def generate_scored_grasp_pairs(self, height, points, cropped_pcd):
        if points.shape[0] == 0:
            return [], []

        points_2d = points[:, :2]

        labels = DBSCAN(eps=0.02, min_samples=1).fit_predict(points_2d)
        main_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
        points_2d = points_2d[labels == main_cluster_label]

        scaler = MinMaxScaler(feature_range=(0, 2))
        scaled_2d_points = scaler.fit_transform(points_2d)

        alpha = 15.0
        alpha_shape = alphashape(scaled_2d_points, alpha)
        while not alpha_shape.geom_type == 'Polygon' and alpha >= 0.0:
            alpha_shape = alphashape(scaled_2d_points, alpha)
            if alpha == 0.0:
                break
            alpha /= 2.0
        if not alpha_shape.geom_type == 'Polygon':
            return [], []
        sorted_points = np.array(alpha_shape.exterior.coords)
        sorted_points = scaler.inverse_transform(sorted_points)

        polygon_points = sorted_points
        if self.debugMode:
            plt.scatter(polygon_points[:, 0], polygon_points[:, 1])
        poly_hole = sg.Polygon(np.array(polygon_points)[::-1])
        straightSkeleton, poly_hole = self.create_skeleton(poly_hole)
        if straightSkeleton is None:
            return [], []

        contour_points, contourGroups = self.contour_point_generation_scored(poly_hole)
        try:
            finalCenters, radiuses, middlesPoints, midllePointRadiuses = self.generateGraspCirclesFromSkeleton(
                poly_hole, straightSkeleton, contour_points)
        except:
            print("invalid Straight Skeleton")
            return [], []
        results, scores = self.keypoint_scored(contour_points, finalCenters, radiuses, cropped_pcd, height,
                                               contourGroups)
        if self.debugMode:
            plt.show()
        return results, scores

    def generate_grasp_poses(self, height_index, point_cloud, step_height=0.01):
        selected_points, cross_heights = self.split_point_cloud_by_height(point_cloud, step_height)

        points = selected_points[height_index]
        if points.shape[0] == 0:
            return []

        points_2d = points[:, :2]
        scaler = MinMaxScaler()
        scaled_2d_points = scaler.fit_transform(points_2d)

        alpha = 9.0
        alpha_shape = alphashape(scaled_2d_points, alpha)
        while not alpha_shape.geom_type == 'Polygon' and alpha >= 0.0:
            alpha_shape = alphashape(scaled_2d_points, alpha)
            alpha -= 1.0
        if not alpha_shape.geom_type == 'Polygon':
            return []
        shape_poly = [alpha_shape]
        sorted_points = np.array(alpha_shape.exterior.coords)
        sorted_points = scaler.inverse_transform(sorted_points)

        polygon_points = sorted_points
        poly_hole = sg.Polygon(np.array(polygon_points)[::-1])
        straightSkeleton = self.create_skeleton(poly_hole)

        contour_points, contourGroups = self.generate_contour_points_from_polygon(poly_hole)
        try:
            finalCenters, radiuses, middlesPoints, midllePointRadiuses = self.generateGraspCirclesFromSkeleton(
                poly_hole, straightSkeleton, contour_points)
        except:
            print("invalid Straight Skeleton")
            return []

        mainGraspPoses = self.keypoint(contour_points, finalCenters, radiuses, contourGroups, shape_poly)
        middleGraspPoses = self.keypoint(contour_points, middlesPoints, midllePointRadiuses, contourGroups,
                                         shape_poly)
        finalGraspPoses = self.filterGraspPosesWithIOU(mainGraspPoses, middleGraspPoses)
        # draw(poly_hole)
        # plt.scatter(points_2d[:,0],points_2d[:,1])
        # for g in mainGraspPoses:
        #     plt.plot([g[0][0],g[1][0]],[g[0][1],g[1][1]])
        # plt.show()
        # plt.close("all")
        if self.debugMode:
            plt.show()
        return finalGraspPoses
