import random
import math

import numpy as np
import pygeos
import pandas as pd
# Smallest enclosing circle - Library (Python)

# Copyright (c) 2017 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program (see COPYING.txt and COPYING.LESSER.txt).
# If not, see <http://www.gnu.org/licenses/>.

# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
#
# Initially: No boundary points known


def _make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not _is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not _is_in_circle(c, q):
            if c[2] == 0.0:
                c = _make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
    circ = _make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if _is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = _make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
            left is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            > _cross_product(px, py, qx, qy, left[0], left[1])
        ):
            left = c
        elif cross < 0.0 and (
            right is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            < _cross_product(px, py, qx, qy, right[0], right[1])
        ):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    if left is None:
        return right
    if right is None:
        return left
    if left[2] <= right[2]:
        return left
    return right


def _make_circumcircle(p0, p1, p2):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
    oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
    ax -= ox
    ay -= oy
    bx -= ox
    by -= oy
    cx -= ox
    cy -= oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = (
        ox
        + (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        )
        / d
    )
    y = (
        oy
        + (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        )
        / d
    )
    ra = math.hypot(x - p0[0], y - p0[1])
    rb = math.hypot(x - p1[0], y - p1[1])
    rc = math.hypot(x - p2[0], y - p2[1])
    return (x, y, max(ra, rb, rc))


def _make_diameter(p0, p1):
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def _is_in_circle(c, p):
    return (
        c is not None
        and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON
    )


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


# end of Nayuiki script to define the smallest enclosing circle


# calculate the area of circumcircle
def _circle_area(points):
    if len(points[0]) == 3:
        points = [x[:2] for x in points]
    circ = _make_circle(points)
    return math.pi * circ[2] ** 2

def _circle_radius(points):
    if len(points[0]) == 3:
        points = [x[:2] for x in points]
    circ = _make_circle(points)
    return circ[2]


def _true_angle(a, b, c):
    # calculate angle between points, return true or false if real corner
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    if np.degrees(angle) <= 170:
        return True
    if np.degrees(angle) >= 190:
        return True
    return False


def get_corners(geom):
#    count corners of geom
    if geom is None:
        return None
    corners = 0  # define empty variables
    points = list(geom.exterior.coords)  # get points of a shape
    stop = len(points) - 1  # define where to stop
    for i in np.arange(
        len(points)
    ):  # for every point, calculate angle and add 1 if True angle
        if i == 0:
            continue
        elif i == stop:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[1])

            if _true_angle(a, b, c) is True:
                corners = corners + 1
            else:
                continue

        else:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[i + 1])

            if _true_angle(a, b, c) is True:
                corners = corners + 1
            else:
                continue
    
    return corners

def squareness(geom):
    if geom is None:
        return None
    
    def _angle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))

        return angle
    
    angles = []
    points = list(geom.exterior.coords)  # get points of a shape
    stop = len(points) - 1  # define where to stop
    for i in np.arange(
        len(points)
    ):  # for every point, calculate angle and add 1 if True angle
        if i == 0:
            continue
        elif i == stop:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[1])
            ang = _angle(a, b, c)

            if ang <= 175:
                angles.append(ang)
            elif _angle(a, b, c) >= 185:
                angles.append(ang)
            else:
                continue

        else:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[i + 1])
            ang = _angle(a, b, c)

            if _angle(a, b, c) <= 175:
                angles.append(ang)
            elif _angle(a, b, c) >= 185:
                angles.append(ang)
            else:
                continue
    deviations = [abs(90 - i) for i in angles]
    return np.mean(deviations)


def elongation(bbox):
    a = bbox.area
    p = bbox.length
    cond1 = p ** 2
    cond2 = 16 * a
    bigger = cond1 >= cond2
    sqrt = np.empty(len(a))
    sqrt[bigger] = cond1[bigger] - cond2[bigger]
    sqrt[~bigger] = 0

    elo1 = ((p - np.sqrt(sqrt)) / 4) / ((p / 2) - ((p - np.sqrt(sqrt)) / 4))
    elo2 = ((p + np.sqrt(sqrt)) / 4) / ((p / 2) - ((p + np.sqrt(sqrt)) / 4))

    # use the smaller one (e.g. shorter/longer)
    res = np.empty(len(a))
    res[elo1 <= elo2] = elo1[elo1 <= elo2]
    res[~(elo1 <= elo2)] = elo2[~(elo1 <= elo2)]
    return res


def centroid_corner(geom):
    '''all these characters working with corners could be merged and cleaned
    '''
    from shapely.geometry import Point
    
    if geom is None:
        return (None, None)
    
    distances = []  # set empty list of distances
    centroid = geom.centroid  # define centroid
    points = list(geom.exterior.coords)  # get points of a shape
    stop = len(points) - 1  # define where to stop
    for i in np.arange(
        len(points)
    ):  # for every point, calculate angle and add 1 if True angle
        if i == 0:
            continue
        elif i == stop:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[1])
            p = Point(points[i])

            if _true_angle(a, b, c) is True:
                distance = centroid.distance(
                    p
                )  # calculate distance point - centroid
                distances.append(distance)  # add distance to the list
            else:
                continue

        else:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[i + 1])
            p = Point(points[i])

            if _true_angle(a, b, c) is True:
                distance = centroid.distance(p)
                distances.append(distance)
            else:
                continue
    if not distances:  # circular buildings
        from momepy.dimension import _longest_axis

        if geom.has_z:
            coords = [
                (coo[0], coo[1]) for coo in geom.convex_hull.exterior.coords
            ]
        else:
            coords = geom.convex_hull.exterior.coords
        return (_longest_axis(coords) / 2, 0)

    return (np.mean(distances), np.std(distances))


def _azimuth(point1, point2):
    """azimuth between 2 shapely points (interval 0 - 180)"""
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def solar_orientation_poly(bbox):
    if bbox is None:
        return None
    bbox = list(bbox.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])
    
    if 90 > az >= 45:
        diff = az - 45
        az = az - 2 * diff
    elif 135 > az >= 90:
        diff = az - 90
        az = az - 2 * diff
        diff = az - 45
        az = az - 2 * diff
    elif 181 > az >= 135:
        diff = az - 135
        az = az - 2 * diff
        diff = az - 90
        az = az - 2 * diff
        diff = az - 45
        az = az - 2 * diff
        
    return az


def street_profile(streets, buildings, distance=3, tick_length=50):

    pygeos_lines = streets.geometry.values.data

    list_points = np.empty((0, 2))
    ids = []

    lengths = pygeos.length(pygeos_lines)
    for ix, (line, length) in enumerate(zip(pygeos_lines, lengths)):

        pts = pygeos.line_interpolate_point(
            line, np.linspace(0, length, num=int((length) // distance))
        )  # .1 offset to keep a gap between two segments
        list_points = np.append(list_points, pygeos.get_coordinates(pts), axis=0)
        ids += [ix] * len(pts) * 2


    ticks = []
    for num, pt in enumerate(list_points, 1):
        # start chainage 0
        if num == 1:
            angle = _getAngle(pt, list_points[num])
            line_end_1 = _getPoint1(pt, angle, tick_length / 2)
            angle = _getAngle(line_end_1, pt)
            line_end_2 = _getPoint2(line_end_1, angle, tick_length)
            ticks.append([line_end_1, pt])
            ticks.append([line_end_2, pt])

        # everything in between
        if num < len(list_points) - 1:
            angle = _getAngle(pt, list_points[num])
            line_end_1 = _getPoint1(
                list_points[num], angle, tick_length / 2
            )
            angle = _getAngle(line_end_1, list_points[num])
            line_end_2 = _getPoint2(line_end_1, angle, tick_length)
            ticks.append([line_end_1, list_points[num]])
            ticks.append([line_end_2, list_points[num]])

        # end chainage
        if num == len(list_points):
            angle = _getAngle(list_points[num - 2], pt)
            line_end_1 = _getPoint1(pt, angle, tick_length / 2)
            angle = _getAngle(line_end_1, pt)
            line_end_2 = _getPoint2(line_end_1, angle, tick_length)
            ticks.append([line_end_1, pt])
            ticks.append([line_end_2, pt])

    ticks = pygeos.linestrings(ticks)
    inp, res = pygeos.STRtree(ticks).query_bulk(buildings.geometry.values.data, predicate='intersects')
    intersections = pygeos.intersection(ticks[res], buildings.geometry.values.data[inp])
    distances = pygeos.distance(intersections, pygeos.points(list_points[res // 2]))

    dists = np.zeros((len(ticks),))
    dists[:] = np.nan
    dists[res] = distances

    ids = np.array(ids)
    widths = []
    openness = []
    deviations = []
    for i in range(len(streets)):
        f = ids == i
        s = dists[f]
        lefts = s[::2]
        rights = s[1::2]
        left_mean = np.nanmean(lefts) if ~np.isnan(lefts).all() else tick_length / 2
        right_mean = np.nanmean(rights) if ~np.isnan(rights).all() else tick_length / 2
        widths.append(np.mean([left_mean, right_mean]) * 2)
        openness.append(np.isnan(s).sum() / (f).sum())
        deviations.append(np.nanstd(s))
    
    return (widths, deviations, openness)


# http://wikicode.wikidot.com/get-angle-of-line-between-two-points
# https://glenbambrick.com/tag/perpendicular/
# angle between two points
def _getAngle(pt1, pt2):
    """
    pt1, pt2 : tuple
    """
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    return math.degrees(math.atan2(y_diff, x_diff))

# start and end points of chainage tick
# get the first end point of a tick
def _getPoint1(pt, bearing, dist):
    """
    pt : tuple
    """
    angle = bearing + 90
    bearing = math.radians(angle)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return (x, y)

# get the second end point of a tick
def _getPoint2(pt, bearing, dist):
    """
    pt : tuple
    """
    bearing = math.radians(bearing)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return (x, y)


def get_edge_ratios(df, edges):
    """
    df: cells/buildngs
    edges: network
    """
    
    # intersection-based join
    buff = edges.buffer(0.01)  # to avoid floating point error
    inp, res = buff.sindex.query_bulk(df.geometry, predicate='intersects')
    intersections = df.iloc[inp].reset_index(drop=True).intersection(buff.iloc[res].reset_index(drop=True))
    mask = intersections.area > 0.0001
    intersections = intersections[mask]
    inp = inp[mask]
    lengths = intersections.area
    grouped = lengths.groupby(inp)
    totals = grouped.sum()
    ints_vect = []
    for name, group in grouped:
        ratios = group / totals.loc[name]
        ints_vect.append({res[item[0]]: item[1] for item in ratios.iteritems()})
    
    edge_dicts = pd.Series(ints_vect, index=totals.index)
    
    # nearest neighbor join
    nans = df.index[~df.index.isin(edge_dicts.index)]
    buffered = df.loc[nans].buffer(500)
    additional = []
    for i in range(len(buffered)):
        geom = buffered.geometry.iloc[i]
        query = edges.sindex.query(geom)
        b = 500
        while query.size == 0:
            query = edges.sindex.query(geom.buffer(b))
            b += 500
        additional.append({edges.iloc[query].distance(geom).idxmin(): 1})

    additional = pd.Series(additional, index=nans)
    return pd.concat([edge_dicts, additional]).sort_index()


def get_nodes(df, nodes, edges, node_id, edge_id, startID, endID):
    nodes = nodes.set_index('nodeID')
    
    node_ids = []

    for edge_dict, geom in zip(df[edge_id], df.geometry):
        edge = edges.iloc[max(edge_dict, key=edge_dict.get)]
        startID = edge.node_start
        start = nodes.loc[startID].geometry
        sd = geom.distance(start)
        endID = edge.node_end
        end = nodes.loc[endID].geometry
        ed = geom.distance(end)
        if sd > ed:
            node_ids.append(endID)
        else:
            node_ids.append(startID)
    
    return pd.Series(node_ids, index=df.index)