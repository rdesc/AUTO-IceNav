import random
from typing import Union, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point

from ship_ice_planner.geometry.utils import euclid_dist

__all__ = [
    'poly_radius',
    'poly_area',
    'poly_centroid',
    'poly_centroid_shapely',
    'generate_polygon',
    'generate_body_points_polygon',
    'shrink_or_swell_polygon',
    'separate_polygons'
]


def poly_radius(vertices, centre_pos) -> float:
    return max(euclid_dist(v, centre_pos) for v in vertices)


def poly_area(vertices):
    # area of solid simple polygon given set of vertices (shape is n x 2)
    # https://en.wikipedia.org/wiki/Polygon#Simple_polygons
    x, y = vertices.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def poly_centroid(vertices):
    # centroid for solid simple polygon given set of vertices (shape is n x 2)
    # https://en.wikipedia.org/wiki/Polygon#Centroid
    assert np.all(vertices >= 0), 'vertices must be in the 1st quadrant'
    x, y = vertices.T
    A = poly_area(vertices)
    u = x * np.roll(y, 1) - np.roll(x, 1) * y
    return np.abs((
        1 / (6 * A) * np.dot(x + np.roll(x, 1), u),
        1 / (6 * A) * np.dot(y + np.roll(y, 1), u)
    ))


def poly_centroid_shapely(vertices):
    centroid = Polygon(vertices).centroid
    return np.array([centroid.x, centroid.y])


def generate_polygon(diameter,
                     origin=(0, 0),
                     num_vertices_range=(10, 20),
                     circular=False,
                     final_vertices_range=None) -> Union[None, np.ndarray]:
    """
    Implements the random convex polygon generation algorithm from
    https://cglab.ca/~sander/misc/ConvexGeneration/convex.html

    :param diameter: diameter of the circle that the polygon is inscribed in
    :param origin: origin of the polygon
    :param num_vertices_range: range of number of vertices in the polygon
    :param circular: if True, generates a circular-like polygon
    :param final_vertices_range: range of number of vertices in the final polygon
    """
    num_vertices = random.randint(*num_vertices_range)

    if circular:
        # for more circular-like obstacles
        x = np.cos(np.linspace(0, 2 * np.pi, num_vertices)) * diameter / 2
        y = np.sin(np.linspace(0, 2 * np.pi, num_vertices)) * diameter / 2

    else:
        # generate two lists of x and y of N random integers between 0 and n
        # sample a random number for number of vertices
        x = [random.uniform(0, diameter) for _ in range(num_vertices)]
        y = [random.uniform(0, diameter) for _ in range(num_vertices)]

    # sort both lists
    x.sort()
    y.sort()

    x_max = x[-1]
    y_max = y[-1]
    x_min = x[0]
    y_min = y[0]

    last_top = x_min
    lastBot = x_min
    x_vec = []

    for i in range(1, num_vertices - 1):
        val = x[i]
        if bool(random.getrandbits(1)):
            x_vec.append(val - last_top)
            last_top = val
        else:
            x_vec.append(lastBot - val)
            lastBot = val

    x_vec.append(x_max - last_top)
    x_vec.append(lastBot - x_max)

    last_left = y_min
    last_right = y_min
    y_vec = []

    for i in range(1, num_vertices - 1):
        val = y[i]
        if bool(random.getrandbits(1)):
            y_vec.append(val - last_left)
            last_left = val
        else:
            y_vec.append(last_right - val)
            last_right = val

    y_vec.append(y_max - last_left)
    y_vec.append(last_right - y_max)
    random.shuffle(y_vec)

    pairs = zip(x_vec, y_vec)
    sorted_pairs = sorted(pairs, key=lambda pair: np.arctan2(pair[0], pair[1]))

    min_polygon_x = 0
    min_polygon_y = 0
    x = 0
    y = 0
    points = []

    for pair in sorted_pairs:
        points.append((x, y))
        x += pair[0]
        y += pair[1]
        min_polygon_x = min(min_polygon_x, x)
        min_polygon_y = min(min_polygon_y, y)

    x_shift = x_min - min_polygon_x
    y_shift = y_min - min_polygon_y

    points = np.asarray(points) + np.array([x_shift, y_shift]).T

    # find the centroid of polygon
    centre_pos = poly_centroid(points)

    # make centre of polygon at the origin
    points -= centre_pos - np.asarray(origin)  # assumes origin is in the 1st quadrant (positive x,y)

    if final_vertices_range is not None:
        # downsample to a random number of vertices
        num_vertices = random.randint(*final_vertices_range)
        if len(points) > num_vertices:
            points = points[::int(len(points) / num_vertices)]

    # n x 2 array where each element is a vertex (x, y)
    return points


def generate_body_points_polygon(
        vertices: Union[List, np.ndarray],
        spacing: float,
        margin=0.,
        weights: np.ndarray = None,
        only_inside_polygon=True,
        plot=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    generates equally spaced points inside polygon

    :param vertices: vertices of polygon
    :param spacing: spacing between points
    :param margin: margin to add the bounds of the sampled points, used for centering the points!
    :param weights: weights for each body point, if array then weights are given
    :param only_inside_polygon: if True, only points inside the polygon are returned
    :param plot: if True, plots the points
    """
    spacing -= 1e-6  # to avoid floating point errors
    polygon = Polygon(vertices)
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    x_coords = np.arange(min_x + margin, max_x - margin, spacing)
    y_coords = np.arange(min_y + margin, max_y - margin, spacing)

    if only_inside_polygon:
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if polygon.contains(point) or polygon.touches(point):
                    points.append((x, y))
    else:
        points = [(x, y) for x in x_coords for y in y_coords]

    points = np.asarray(points)

    if weights is None:
        # generate weights for each vertex
        weights = np.ones(len(points))
        # find the min number of vertices in a row
        row_sizes = {y: len(points[points[:, 1] == y]) for y in y_coords}
        min_row_size = min([item for item in row_sizes.values() if item != 0])

        for idx, item in enumerate(points):
            weights[idx] = min_row_size / row_sizes[item[1]]

    if plot:
        print('number of points:', len(points))
        print('number of rows:', len(y_coords), 'number of cols:', len(x_coords))
        print('weights', list(weights))

        x, y = polygon.exterior.xy
        f, ax = plt.subplots()
        ax.plot(x, y, 'k')
        ax.scatter(*points.T, c=weights)
        # show the index of each point
        for i, p in enumerate(points):
            plt.text(p[0], p[1], str(i), fontsize=8)
        ax.add_patch(plt.Polygon(vertices, True, fill=False, linewidth=2))
        ax.set_aspect('equal')
        plt.show()

    return points, weights


def shrink_or_swell_polygon(vertices, factor=0.10, swell=False, plot=False):
    """
    returns the shapely polygon which is smaller or bigger by passed factor.
    If swell = True , then it returns bigger polygon, else smaller

    Original code from https://stackoverflow.com/a/67205583
    """
    polygon = Polygon(vertices)
    polygon_resized = _shrink_swell_helper(polygon, factor, swell)

    if polygon_resized.area <= 0:
        return vertices  # return original

    if type(polygon_resized) is not Polygon:
        return vertices  # return original

    if plot:
        # visualize for debugging
        x, y = polygon.exterior.xy
        plt.plot(x, y)
        x, y = polygon_resized.exterior.xy
        plt.plot(x, y)
        plt.axis('equal')
        plt.show()

    return np.asarray(polygon_resized.exterior.xy).T


def _shrink_swell_helper(polygon: Polygon, factor: float, swell: bool = False) -> Polygon:
    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * factor

    if swell:
        polygon_resized = polygon.buffer(shrink_distance)  # expand
    else:
        polygon_resized = polygon.buffer(-shrink_distance)  # shrink

    return polygon_resized


def separate_polygons(vertices: Union[List, np.ndarray],
                      factor=0.10,
                      smoothen_corners=True,
                      filter_out_small_polygons=True,
                      min_ob_area=0) -> List[np.ndarray]:
    # simple fix to split 'easy' non-convex polygons, e.g. two polygons connected by a point
    #   _____  _________
    #  /     \/         \
    #  |     /\         |
    #  \____/  \________/
    polygon = Polygon(vertices)
    initial_area = polygon.area

    if initial_area <= min_ob_area:
        if filter_out_small_polygons:
            return []
        return [vertices]

    polygon_resized = _shrink_swell_helper(polygon, factor=factor)

    if polygon_resized.area <= min_ob_area:
        if filter_out_small_polygons:
            return []
        return [vertices]

    grow_factor = initial_area / polygon_resized.area  # this factor can explode if resized polygon is too small

    if type(polygon_resized) is not Polygon:
        new_polygons = []
        for poly in polygon_resized.geoms:
            new_poly = _shrink_swell_helper(poly, factor=grow_factor * factor, swell=True)
            if type(new_poly) is Polygon:
                new_polygons.append(
                    np.asarray(new_poly.exterior.xy).T
                )
        return new_polygons

    if smoothen_corners:
        try:
            return [np.asarray(_shrink_swell_helper(polygon_resized, factor=grow_factor * factor, swell=True).exterior.xy).T]
        except AttributeError as e:
            print('Weird error:', e, 'simply returning the original polygon...')
            return [vertices]

    return [vertices]
