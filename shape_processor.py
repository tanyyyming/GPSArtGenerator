import numpy as np
import cv2 as cv
from rdp import rdp
from scipy.spatial.distance import cdist
from collections import defaultdict

# Directions for 8 surrounded neighbor pixels in a 2D grid
DIRS_DEFAULT = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
# Directions for 8 neighbors but with priority for the first quadrant
DIRS_PRIOR_Q1 = [
    # first quadrant
    (1, 1),
    (1, 0),
    (0, 1),
    # other quadrants with diagonal directions first
    (1, -1),
    (-1, 1),
    (0, -1),
    (-1, 0),
    # fourth quadrant the last as it is the incoming direction
    (-1, -1),
]
DIRS_PRIOR_Q2 = [
    (-1, 1),
    (-1, 0),
    (0, 1),
    (-1, -1),
    (1, 1),
    (0, -1),
    (1, 0),
    (1, -1),
]
DIRS_PRIOR_Q3 = [
    (-1, -1),
    (-1, 0),
    (0, -1),
    (1, -1),
    (-1, 1),
    (0, 1),
    (1, 0),
    (1, 1),
]
DIRS_PRIOR_Q4 = [
    (1, -1),
    (1, 0),
    (0, -1),
    (-1, -1),
    (1, 1),
    (0, 1),
    (-1, 0),
    (-1, 1),
]


def generate_neighbors(directions, point) -> list[tuple[int, int]]:
    return [(point[0] + dx, point[1] + dy) for dx, dy in directions]


def has_immediate_neighbor(point, points_set, except_point):
    for neighbor in generate_neighbors(DIRS_DEFAULT, point):
        if neighbor in points_set and np.all(neighbor != except_point):
            return True
    return False


def get_arc_length(points: np.ndarray) -> float:
    arc_length = np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
    return arc_length


def convert_to_points(im) -> np.ndarray:
    # Convert to grayscale
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", imgray)
    cv.waitKey(0)

    # Threshold the image to black and white
    _, imthresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV)
    cv.imshow("thresh", imthresh)
    cv.waitKey(0)

    # Thinning the image to make line segments 1 pixel wide (in general)
    imthinned = cv.ximgproc.thinning(imthresh)
    cv.imshow("thinned", imthinned)
    cv.waitKey(0)

    # Find the coordinates of the white pixels
    points = cv.findNonZero(imthinned)

    if points is not None:
        points = points.reshape(-1, 2)  # Reshape to Nx2 array of (x, y) coordinates

    return points


def get_line_trace(points: np.ndarray) -> np.ndarray:
    points_set = set(map(tuple, points))

    num_neighbors_to_points = defaultdict(list)
    for point in points:
        count = sum(
            1
            for neighbor in generate_neighbors(DIRS_DEFAULT, point)
            if neighbor in points_set
        )
        num_neighbors_to_points[count].append(point)
    print(
        f"num_neighbors_to_points: { {k: len(v) for k, v in num_neighbors_to_points.items()} }"
    )
    # possible start points of the line trace
    single_neighbor_points = num_neighbors_to_points[1]

    if not single_neighbor_points:
        print("No single neighbor points found")
        # a closed line, so any point can be a start point
        return sort_points_by_proximity(
            points_set.copy(), points[0], append_start_point=True
        )

    min_avg_arc_length = np.inf
    res = None
    # the one with the smallest average arc length is the best candidate as
    # the start point of the line trace because there's no large jumps
    # between the points
    for point in single_neighbor_points:
        sorted_points = sort_points_by_proximity(points_set.copy(), point)
        avg_arc_length = get_arc_length(sorted_points) / len(sorted_points)
        print(
            f"num_points: {len(sorted_points)},"
            f"total_arc_length: {get_arc_length(sorted_points)},"
            f"average_arc_length: {avg_arc_length}"
        )
        if avg_arc_length < min_avg_arc_length:
            min_avg_arc_length = avg_arc_length
            res = sorted_points
    return res


def sort_points_by_proximity(
    points_set: set[tuple[int, int]], start_point, append_start_point=False
) -> np.ndarray:
    current_point = tuple(start_point)

    sorted_points = [current_point]
    has_old_neighbor = [False]
    visited = {current_point}

    while len(visited) < len(points_set):
        # Determine the search sequence based on the previous point
        # Heuristics: search for the next best point in continuation of the previous direction
        if len(sorted_points) < 2:
            search_sequence = DIRS_DEFAULT
        else:
            prev_point = sorted_points[-2]
            xdiff = current_point[0] - prev_point[0]
            ydiff = current_point[1] - prev_point[1]
            if xdiff >= 0 and ydiff >= 0:
                search_sequence = DIRS_PRIOR_Q1
            elif xdiff < 0 and ydiff >= 0:
                search_sequence = DIRS_PRIOR_Q2
            elif xdiff < 0 and ydiff < 0:
                search_sequence = DIRS_PRIOR_Q3
            else:
                search_sequence = DIRS_PRIOR_Q4

        next_point = None
        has_new_neighbor = False
        for neighbor in generate_neighbors(search_sequence, current_point):
            if neighbor in points_set and neighbor not in visited:
                # If a neighbor is found, move to it
                has_new_neighbor = True
                next_point = neighbor
                break

        if not has_new_neighbor:
            remaining_points = [
                point for point in list(points_set) if point not in visited
            ]
            distances = cdist([current_point], remaining_points)
            assert next_point is None
            next_point = tuple(remaining_points[np.argmin(distances)])

        has_old_neighbor.append(
            has_immediate_neighbor(next_point, visited, current_point)
        )
        sorted_points.append(next_point)
        visited.add(next_point)
        current_point = next_point

    # Heuristics: the total number of points extracted from cv is greater than
    # the number of points required to trace the line. Many leftover points
    # are dangled at the end that should be removed. The way to see if they
    # are necessary is to check if they have any neighbors that are already
    # visited far back in the trace (obtained from has_old_neighbor[]).
    # If they do, then they are not necessary.
    end_index = len(sorted_points) - has_old_neighbor[::-1].index(False)
    sorted_points = sorted_points[:end_index]

    if append_start_point:
        sorted_points.append(sorted_points[0])

    return np.array(sorted_points)


def rdp_simplify(points, epsilon=0.01):
    return rdp(points, epsilon * get_arc_length(points))


def convert_to_coordinates(
    points: np.ndarray, image_dimensions: tuple[int, int], bbox: list[float]
) -> np.ndarray:
    """
    Convert the points to coordinates in the given bounding box.

    Note that the points has its origin at the top-left corner of the image.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    h, w = image_dimensions
    ratio = min(lat_range / h, lon_range / w)

    # While points' y coordinates increase downwards, latitudes increase upwards
    # So, we need to flip the y coordinates
    lat_points = max_lat - points[:, 1] * ratio
    lon_points = points[:, 0] * ratio + min_lon
    return np.stack((lat_points, lon_points), axis=1)


def line_trace_to_coordinates(
    points: np.ndarray, image_dimensions: tuple[int, int], bbox: list[float]
) -> np.ndarray:
    simplified_points = rdp_simplify(points)
    return convert_to_coordinates(simplified_points, image_dimensions, bbox)


def image_to_coordinates(image_filename: str, bbox: list[float]) -> np.ndarray:
    im = cv.imread(image_filename)
    assert im is not None, "file could not be read, check with os.path.exists()"
    h, w, _ = im.shape
    points = convert_to_points(im)
    sorted_points = get_line_trace(points)
    simplified_points = rdp_simplify(sorted_points)

    for point in simplified_points:
        cv.circle(im, tuple(point), 2, (0, 255, 0), -1)
    cv.imshow("Simplified Points", im)
    cv.waitKey(0)

    return convert_to_coordinates(simplified_points, (h, w), bbox)


# skeleton = np.zeros_like(imthresh)
# element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
# while True:
#     eroded = cv.erode(imthresh, element)
#     temp = cv.dilate(eroded, element)
#     temp = cv.subtract(imthresh, temp)
#     skeleton = cv.bitwise_or(skeleton, temp)
#     imthresh = eroded.copy()
#     if cv.countNonZero(imthresh) == 0:
#         break
# cv.imshow('skeleton', skeleton)
# cv.waitKey(0)

# contours, _ = cv.findContours(thinned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(im, contours, 0, (0, 255, 0), 3)
# cv.imshow('contours', im)
# cv.waitKey(0)

# contour = contours[0]
# epsilon = 0.005 * cv.arcLength(contour, True)  # Adjust epsilon for simplification level
# simplified_contour = cv.approxPolyDP(contour, epsilon, True)
# cv.drawContours(im, [simplified_contour], 0, (0, 0, 255), 2)
# cv.imshow('simplified', im)
# cv.waitKey(0)
