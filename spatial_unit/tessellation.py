import dask.bag as db
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos
from scipy.spatial import Voronoi
from shapely.ops import polygonize


class Tessellation:
    """
    Taken from momepy, updated version will go back.

    Generate morphological tessellation around given buildings or proximity bands around
    given street network.

    See :cite:`fleischmann2020` for details of implementation.

    Tessellation requires data of relatively high level of precision and there are three
    particular patterns causign issues.\n
    1. Features will collapse into empty polygon - these do not have tessellation
    cell in the end.\n
    2. Features will split into MultiPolygon - at some cases, features with narrow links
    between parts split into two during 'shrinking'. In most cases that is not an issue
    and resulting tessellation is correct anyway, but sometimes this result in a cell
    being MultiPolygon, which is not correct.\n
    3. Overlapping features - features which overlap even after 'shrinking' cause invalid
    tessellation geometry.\n
    All three types can be tested prior :class:`momepy.Tessellation` using :class:`momepy.CheckTessellationInput`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing building footprints or street network
    unique_id : str
        name of the column with unique id
    limit : MultiPolygon or Polygon
        MultiPolygon or Polygon defining the study area limiting tessellation (otherwise it could go to infinity).
    shrink : float (default 0.4)
        distance for negative buffer to generate space between adjacent polygons (if geometry type of gdf is (Multi)Polygon).
    segment : float (default 0.5)
        maximum distance between points after discretization
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    tessellation : GeoDataFrame
        GeoDataFrame containing resulting tessellation
    gdf : GeoDataFrame
        original GeoDataFrame
    id : Series
        Series containing used unique ID
    limit : MultiPolygon or Polygon
        limit
    shrink : float
        used shrink value
    segment : float
        used segment value
    sindex : rtree spatial index
        spatial index of tessellation
    collapsed : list
        list of unique_id's of collapsed features (if there are some)
    multipolygons : list
        list of unique_id's of features causing MultiPolygons (if there are some)


    Examples
    --------
    >>> tess = mm.Tessellation(buildings_df, 'uID', limit=mm.buffered_limit(buildings_df))
    Inward offset...
    Discretization...
    Generating input point array...
    100%|██████████| 144/144 [00:00<00:00, 376.15it/s]
    Generating Voronoi diagram...
    Generating GeoDataFrame...
    Vertices to Polygons: 100%|██████████| 33059/33059 [00:01<00:00, 31532.72it/s]
    Dissolving Voronoi polygons...
    Preparing buffer zone for edge resolving...
    Building R-tree...
    100%|██████████| 42/42 [00:00<00:00, 752.54it/s]
    Cutting...
    >>> tess.tessellation.head()
        uID	geometry
    0	1	POLYGON ((1603586.677274485 6464344.667944215,...
    1	2	POLYGON ((1603048.399497852 6464176.180701573,...
    2	3	POLYGON ((1603071.342637536 6464158.863329805,...
    3	4	POLYGON ((1603055.834005827 6464093.614718676,...
    4	5	POLYGON ((1603106.417554705 6464130.215958447,...

    Notes
    -------
    queen_corners is currently experimental method only and can cause errors.
    """

    def __init__(
        self, gdf, unique_id, limit, shrink=0.4, segment=0.5, verbose=True,
    ):
        self.gdf = gdf
        self.id = gdf[unique_id]
        self.limit = limit
        self.shrink = shrink
        self.segment = segment

        objects = gdf.copy()

        bounds = pygeos.bounds(limit)
        centre_x = (bounds[0] + bounds[2]) / 2
        centre_y = (bounds[1] + bounds[3]) / 2
        objects["geometry"] = objects["geometry"].translate(
            xoff=-centre_x, yoff=-centre_y
        )

        if shrink != 0:
            print("Inward offset...") if verbose else None
            mask = objects.type.isin(["Polygon", "MultiPolygon"])
            objects.loc[mask, "geometry"] = objects[mask].buffer(
                -shrink, cap_style=2, join_style=2
            )

        objects = objects.reset_index(drop=True).explode()
        objects = objects.set_index(unique_id)

        print("Generating input point array...") if verbose else None
        points, ids = self._dense_point_array(
            objects.geometry.values.data, distance=segment, index=objects.index
        )

        # add convex hull buffered large distance to eliminate infinity issues
        series = gpd.GeoSeries(limit, crs=gdf.crs).translate(
            xoff=-centre_x, yoff=-centre_y
        )
        width = bounds[2] - bounds[0]
        leng = bounds[3] - bounds[1]
        hull = series.geometry[[0]].buffer(width if width > leng else leng)
        hull_p, hull_ix = self._dense_point_array(
            hull.values.data, distance=pygeos.length(limit) / 100, index=hull.index
        )
        points = np.append(points, hull_p, axis=0)
        ids = ids + ([-1] * len(hull_ix))

        print("Generating Voronoi diagram...") if verbose else None
        voronoi_diagram = Voronoi(np.array(points))

        print("Generating GeoDataFrame...") if verbose else None
        regions_gdf = self._regions(voronoi_diagram, unique_id, ids, crs=gdf.crs)

        print("Dissolving Voronoi polygons...") if verbose else None
        morphological_tessellation = regions_gdf[[unique_id, "geometry"]].dissolve(
            by=unique_id, as_index=False
        )

        morphological_tessellation = gpd.clip(morphological_tessellation, series)

        morphological_tessellation["geometry"] = morphological_tessellation[
            "geometry"
        ].translate(xoff=centre_x, yoff=centre_y)

        self.tessellation = morphological_tessellation

    def _dense_point_array(self, geoms, distance, index):
        """
        geoms - array of pygeos lines
        """
        # interpolate lines to represent them as points for Voronoi
        points = np.empty((0, 2))
        ids = []

        lines = pygeos.boundary(geoms)
        lengths = pygeos.length(lines)
        for ix, line, length in zip(index, lines, lengths):
            pts = pygeos.line_interpolate_point(
                line,
                np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance)),
            )  # .1 offset to keep a gap between two segments
            points = np.append(points, pygeos.get_coordinates(pts), axis=0)
            ids += [ix] * len(pts)

        return points, ids

        # here we might also want to append original coordinates of each line
        # to get a higher precision on the corners

    def _regions(self, voronoi_diagram, unique_id, ids, crs):
        """
        Generate GeoDataFrame of Voronoi regions from scipy.spatial.Voronoi.
        """
        vertices = pd.Series(voronoi_diagram.regions).take(voronoi_diagram.point_region)
        polygons = []
        for region in vertices:
            if -1 not in region:
                polygons.append(pygeos.polygons(voronoi_diagram.vertices[region]))
            else:
                polygons.append(None)

        regions_gdf = gpd.GeoDataFrame(
            {unique_id: ids}, geometry=polygons, crs=crs
        ).dropna()
        regions_gdf = regions_gdf.loc[
            regions_gdf[unique_id] != -1
        ]  # delete hull-based cells

        return regions_gdf


def enclosed_tessellation(
    barriers, buildings, limit, unique_id="uID", enclosure_id="eID", enclosures=None
):
    """Enclosed tessellation

    Generate enclosed tessellation based on barriers defining enclosures and buildings
    footprints.

    Production version will be implemented in momepy.

    Parameters
    ----------
    barriers : GeoDataFrame
        GeoDataFrame containing barriers (e.g. street network). Expects (Multi)LineString geometry.
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints. Expects (Multi)Polygon geometry.
    unique_id : str
        name of the column with unique id of buildings gdf
    limit : MultiPolygon or Polygon
        MultiPolygon or Polygon defining the study area limiting tessellation (otherwise it could go to infinity).
    enclosures : geopandas.GeometryArray (optional)
        You can pass GeometryArray of polygonized barriers (as pygeos geometries) to skip potentially expensive
        polygonization.

    Returns
    -------
    tessellation : GeoDataFrame
        gdf contains three columns: 
            geometry,
            unique_id matching with parental building, 
            enclosure_id matching with enclosure index
    """

    # get barrier-based polygons (enclosures)
    if not enclosures:
        polygons = polygonize(
            barriers.geometry.append(gpd.GeoSeries([limit.boundary])).unary_union
        )
        enclosures = gpd.array.from_shapely(list(polygons), crs=barriers.crs)

    # determine which polygons should be split
    inp, res = buildings.sindex.query_bulk(enclosures.data, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    splits = unique[counts > 1]
    single = unique[counts == 1]

    # initialize dask.bag
    bag = db.from_sequence(splits)

    # use enclosed tessellation algorithm
    def tess(
        ix,
        enclosure,
        buildings,
        query_inp,
        query_res,
        threshold=0.05,
        unique_id=unique_id,
        **kwargs
    ):
        poly = enclosure.data[ix]
        blg = buildings.iloc[query_res[query_inp == ix]]
        within = blg[
            pygeos.area(pygeos.intersection(blg.geometry.values.data, poly))
            > (pygeos.area(blg.geometry.values.data) * threshold)
        ]
        tess = Tessellation(within, unique_id, poly, **kwargs)
        tess.tessellation[enclosure_id] = ix
        return tess.tessellation

    # generate enclosed tessellation using dask
    new = bag.map(
        tess, enclosures, buildings, inp, res, shrink=0, segment=2, verbose=False
    ).compute()

    # finalise the result
    clean_blocks = gpd.GeoDataFrame(geometry=enclosures)
    clean_blocks[enclosure_id] = range(len(enclosures))
    clean_blocks = clean_blocks.drop(splits)
    clean_blocks.loc[single, "uID"] = clean_blocks.loc[single].eID.apply(
        lambda ix: buildings.iloc[res[inp == ix][0]][unique_id]
    )
    tessellation = pd.concat(new)

    return tessellation.append(clean_blocks)
