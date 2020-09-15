from itertools import combinations
import collections

import pygeos
import numpy as np
import pandas as pd
import geopandas as gpd
import momepy as mm

from shapely.ops import polygonize
from scipy.spatial import Voronoi


# helper functions
def get_ids(x, ids):
    return ids[x]


mp = np.vectorize(get_ids, excluded=["ids"])


def dist(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_verts(x, voronoi_diagram):
    return voronoi_diagram.vertices[x]


def _average_geometry(lines, poly=None, distance=2):
    """
    Returns average geometry.


    Parameters
    ----------
    lines : list
        LineStrings connected at endpoints forming a closed polygon
    poly : shapely.geometry.Polygon
        polygon enclosed by `lines`
    distance : float
        distance for interpolation

    Returns list of averaged geometries
    """
    if not poly:
        polygons = list(polygonize(lines))
        if len(polygons) == 1:
            poly = polygons[0]
        else:
            raise ValueError("given lines do not form a single polygon")
    # get an additional line around the lines to avoid infinity issues with Voronoi
    extended_lines = [poly.buffer(distance).exterior] + lines

    # interpolate lines to represent them as points for Voronoi
    points = np.empty((0, 2))
    ids = []

    pygeos_lines = pygeos.from_shapely(extended_lines)
    lengths = pygeos.length(pygeos_lines)
    for ix, (line, length) in enumerate(zip(pygeos_lines, lengths)):
        pts = pygeos.line_interpolate_point(
            line, np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance))
        )  # .1 offset to keep a gap between two segments
        points = np.append(points, pygeos.get_coordinates(pts), axis=0)
        ids += [ix] * len(pts)

        # here we might also want to append original coordinates of each line
        # to get a higher precision on the corners, but it does not seem to be
        # necessary based on my tests.

    # generate Voronoi diagram
    voronoi_diagram = Voronoi(points)

    # get all rigdes and filter only those between the two lines
    pts = voronoi_diagram.ridge_points
    mapped = mp(pts, ids=ids)

    # iterate over segment-pairs
    edgelines = []
    for a, b in combinations(range(1, len(lines) + 1), 2):
        mask = (
            np.isin(mapped[:, 0], [a, b])
            & np.isin(mapped[:, 1], [a, b])
            & (mapped[:, 0] != mapped[:, 1])
        )
        rigde_vertices = np.array(voronoi_diagram.ridge_vertices)
        verts = rigde_vertices[mask]

        # generate the line in between the lines
        edgeline = pygeos.line_merge(
            pygeos.multilinestrings(get_verts(verts, voronoi_diagram))
        )
        snapped = pygeos.snap(edgeline, pygeos_lines[a], distance)
        edgelines.append(snapped)
    return edgelines


def consolidate(network, distance=2, epsilon=2, filter_func=None, **kwargs):
    """
    Consolidate edges of a network, takes care of geometry only. No
    attributes are preserved at the moment.

    The whole process is split into several steps:
    1. Polygonize network
    2. Find polygons which are likely caused by dual lines and other
       geometries to be consolidated.
    3. Iterate over those polygons and generate averaged geometry
    4. Remove invalid and merge together with new geometry.

    Step 2 needs work, this is just a first attempt based on shape and area
    of the polygon. We will have to come with clever options here and
    allow their specification, because each network will need different
    parameters.

    Either before or after these steps needs to be done node consolidation,
    but in a way which does not generate overlapping geometries.
    Overlapping geometries cause (unresolvable) issues with Voronoi.

    Parameters
    ----------
    network : GeoDataFrame (LineStrings)

    distance : float
        distance for interpolation

    epsilon : float
        tolerance for simplification

    filter_func : function
        function which takes gdf of polygonized network and returns mask of invalid
        polygons (those which should be consolidated)

    **kwargs
        Additional kwargs passed to filter_func
    """

    # polygonize network
    polygonized = polygonize(network.geometry)
    geoms = [g for g in polygonized]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=network.crs)

    # filter potentially incorrect polygons
    mask = filter_func(gdf, **kwargs)
    invalid = gdf.loc[mask]

    sindex = network.sindex

    # iterate over polygons which are marked to be consolidated
    # list segments to be removed and the averaged geoms replacing them
    averaged = []
    to_remove = []
    for poly in invalid.geometry:
        real = network.iloc[sindex.query(poly.exterior, predicate="intersects")]
        mask = real.intersection(poly.exterior).type.isin(
            ["LineString", "MultiLineString"]
        )
        real = real[mask]
        lines = list(real.geometry)
        to_remove += list(real.index)

        if lines:
            av = _average_geometry(lines, poly, distance)
            averaged += av

    # drop double lines
    clean = network.drop(set(to_remove))

    # merge new geometries with the existing network
    averaged = gpd.array.from_shapely(averaged, crs=network.crs).simplify(epsilon)
    result = pd.concat([clean, gpd.GeoDataFrame(geometry=averaged[~averaged.is_empty])])
    merge = topology(result)

    return merge


def roundabouts(gdf, area=5000, circom=0.6):
    """
    Filter out roundabouts
    """

    # calculate parameters
    gdf["area"] = gdf.geometry.area
    gdf["circom"] = mm.CircularCompactness(gdf, "area").series
    # select valid and invalid network-net_blocks
    mask = (gdf["area"] < area) & (gdf["circom"] > circom)
    return mask


def filter_comp(gdf, max_size=10000, circom_max=0.2):
    """
    Filter based on max size and compactness

    Parameters
    ----------
    gdf : GeoDataFrame
        polygonized network
    max_size : float
        maximum size of a polygon to be considered potentially invalid
    circom_max : float
        maximum circular compactness of a polygon to be considered
        potentially invalid.

    Returns boolean series

    """
    # calculate parameters
    gdf["area"] = gdf.geometry.area
    gdf["circom"] = mm.CircularCompactness(gdf, "area").series
    # select valid and invalid network-net_blocks
    mask = (gdf["area"] < max_size) & (gdf["circom"] < circom_max)
    return mask


def topology(gdf):
    """
    Clean topology of existing LineString geometry by removal of nodes of degree 2.

    Parameters
    ----------
    gdf : GeoDataFrame
        (Multi)LineString data of street network
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
    merge = res[np.isin(inp, unique[counts == 2])]

    # filter duplications and create a dictionary with indication of components to be merged together
    dups = [item for item, count in collections.Counter(merge).items() if count > 1]
    split = np.split(merge, len(merge) / 2)
    components = {}
    for i, a in enumerate(split):
        if a[0] in dups or a[1] in dups:
            if a[0] in components.keys():
                i = components[a[0]]
            elif a[1] in components.keys():
                i = components[a[1]]
        components[a[0]] = i
        components[a[1]] = i

    # iterate through components and create new geometries
    new = []
    for c in set(components.values()):
        keys = []
        for item in components.items():
            if item[1] == c:
                keys.append(item[0])
        new.append(pygeos.line_merge(pygeos.union_all(geom[keys])))

    # remove incorrect geometries and append fixed versions
    df = df.drop(merge)
    final = gpd.GeoSeries(new).explode().reset_index(drop=True)
    return df.append(
        gpd.GeoDataFrame({df.geometry.name: final}, geometry=df.geometry.name),
        ignore_index=True,
    )


def measure_network(xy, user, pwd, host, port, buffer, area, circom, cons=True):
    import networkx as nx
    from sqlalchemy import create_engine

    url = f"postgres+psycopg2://{user}:{pwd}@{host}:{port}/built_env"
    engine = create_engine(url)

    sql = f"SELECT * FROM openroads_200803_topological WHERE ST_DWithin(geometry, ST_SetSRID(ST_Point({xy[0][0]}, {xy[0][1]}), 27700), {buffer})"

    df = gpd.read_postgis(sql, engine, geom_col="geometry")

    try:
        if cons:
            topo = consolidate(df, filter_func=roundabouts, area=area, circom=circom)
        else:
            topo = df
        G = mm.gdf_to_nx(topo)
        mesh = mm.meshedness(G, radius=None)
        G = mm.subgraph(
            G,
            meshedness=True,
            cds_length=False,
            mean_node_degree=False,
            proportion={0: False, 3: False, 4: False},
            cyclomatic=False,
            edge_node_ratio=False,
            gamma=False,
            local_closeness=True,
            closeness_weight=None,
            verbose=False,
        )
        vals = list(nx.get_node_attributes(G, "meshedness").values())
        l_mesh_mean = np.mean(vals)
        l_mesh_median = np.median(vals)
        l_mesh_dev = np.std(vals)
        vals = list(nx.get_node_attributes(G, "local_closeness").values())
        l_close_mean = np.mean(vals)
        l_close_median = np.median(vals)
        l_close_dev = np.std(vals)

        return [
            mesh,
            l_mesh_mean,
            l_mesh_median,
            l_mesh_dev,
            l_close_mean,
            l_close_median,
            l_close_dev,
        ]
    except ValueError:
        return None
