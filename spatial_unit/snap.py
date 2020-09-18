import math
import numpy as np
import pygeos
import operator
import geopandas as gpd


def get_extrapolated_line(coords, tolerance, point=False):
    """
    Creates a line extrapoled in p1->p2 direction.
    """
    p1 = coords[:2]
    p2 = coords[2:]
    a = p2

    # defining new point based on the vector between existing points
    if p1[0] >= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] <= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    else:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    if point:
        return b
    return pygeos.linestrings([a, b])


def line_to_line(gdf, target, tolerance):
    """ Extends lines from gdf to target within a set tolerance
   
    """
    # explode to avoid MultiLineStrings
    # double reset index due to the bug in GeoPandas explode
    df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

    # get underlying pygeos geometry
    geom = df.geometry.values.data

    # extract array of coordinates and number per geometry
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = pygeos.STRtree(geom)
    inp, res = tree.query_bulk(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    new_geoms = []
    # iterate over cul-de-sac-like segments and attempt to snap them to street network
    for line in ends:

        l_coords = pygeos.get_coordinates(geom[line])

        start = pygeos.points(l_coords[0])
        end = pygeos.points(l_coords[-1])

        first = list(tree.query(start, predicate="intersects"))
        second = list(tree.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        if first and not second:
            snapped = extend_line(l_coords, target, tolerance)
            new_geoms.append(
                pygeos.linestrings(extend_line(snapped, target, 0.00001, snap=False))
            )
        elif not first and second:
            snapped = extend_line(np.flip(l_coords, axis=0), target, tolerance)
            new_geoms.append(
                pygeos.linestrings(extend_line(snapped, target, 0.00001, snap=False))
            )
        elif not first and not second:
            one_side = extend_line(l_coords, target, tolerance)
            one_side_e = extend_line(one_side, target, 0.00001, snap=False)
            snapped = extend_line(np.flip(one_side_e, axis=0), target, tolerance)
            new_geoms.append(
                pygeos.linestrings(extend_line(snapped, target, 0.00001, snap=False))
            )

    df = df.drop(ends)
    final = gpd.GeoSeries(new_geoms).explode().reset_index(drop=True)
    return df.append(
        gpd.GeoDataFrame({df.geometry.name: final}, geometry=df.geometry.name),
        ignore_index=True,
    )


def extend_line(coords, target, tolerance, snap=True):
    """
    Extends a line geometry to snap on the target within a tolerance.
    """
    if snap:
        extrapolation = get_extrapolated_line(
            coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(), tolerance
        )
        int_idx = target.sindex.query(extrapolation, predicate="intersects")
        intersection = pygeos.intersection(
            target.iloc[int_idx].geometry.values.data, extrapolation
        )
        if intersection.size > 0:
            if len(intersection) > 1:
                distances = {}
                ix = 0
                for p in intersection:
                    distance = pygeos.distance(p, pygeos.points(coords[-1]))
                    distances[ix] = distance
                    ix = ix + 1
                minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                new_point_coords = pygeos.get_coordinates(intersection[minimal])

            else:
                new_point_coords = pygeos.get_coordinates(intersection[0])
            coo = np.append(coords, new_point_coords)
            new = np.reshape(coo, (int(len(coo) / 2), 2))

            return new
        return coords

    extrapolation = get_extrapolated_line(
        coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
        tolerance,
        point=True,
    )
    return np.vstack([coords, extrapolation])


def close_gaps(df, tolerance):
    """Close gaps in LineString geometry where it should be contiguous.

    Snaps both lines to a centroid of a gap in between.

    """
    geom = df.geometry.values.data
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    buffered = pygeos.buffer(points, tolerance)

    dissolved = pygeos.union_all(buffered)

    exploded = [
        pygeos.get_geometry(dissolved, i)
        for i in range(pygeos.get_num_geometries(dissolved))
    ]

    centroids = pygeos.centroid(exploded)

    snapped = pygeos.snap(geom, pygeos.union_all(centroids), tolerance)

    return snapped
