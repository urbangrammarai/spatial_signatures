import numpy as np
import geopandas as gpd


def dask_dissolve(ddf, by=None, aggfunc="first"):
    """
    ddf should be sorted according to `by` for the best performance
    
    n_partitoins should be approximately workers * 4
    
    In case geometry columns is no called "geometry" it needs to be reassigned after `.compute()`
    """    
    def _partition_dissolve(partition, by=None, aggfunc="first", geometry_name="geometry"):
        
        if by is None:
            by = np.zeros(len(partition), dtype="int64")

        # Process non-spatial component
        data = partition.drop(labels=geometry_name, axis=1)
        aggregated_data = data.groupby(by=by).agg(aggfunc)

        # Process spatial component
        def merge_geometries(block):
            merged_geom = gpd.GeoSeries(block).unary_union
            return merged_geom

        g = partition.groupby(by=by, group_keys=False)[geometry_name].agg(
            merge_geometries
        )

        # Aggregate
        aggregated_geometry = gpd.GeoDataFrame(g, geometry=geometry_name, crs=27700)
        # Recombine
        aggregated = aggregated_geometry.join(aggregated_data)

        aggregated = aggregated.reset_index()

        return aggregated
    
    geometry_name = ddf.geometry.name
    cols = ddf.columns.to_list()
    cols.remove(geometry_name)
    
    if by is None:
        cols.insert(0, 'index')
        cols.insert(1, geometry_name)
    else:
        cols.remove(by)
        cols.insert(0, by)
        cols.insert(1, geometry_name)
    
    first_step = ddf.map_partitions(_partition_dissolve, by=by, geometry_name=geometry_name, aggfunc=aggfunc, meta=gpd.GeoDataFrame(columns=cols))
    
    if by is None:
        pre_second = first_step.set_index('index')
    else:
        pre_second = first_step.set_index(by)
        
    un = ddf[by].unique().compute()
    repartiotioned = pre_second.repartition(divisions=sorted(list(un)))
    
    second_step = repartiotioned.map_partitions(_partition_dissolve, by=by, geometry_name=geometry_name, aggfunc=aggfunc, meta=gpd.GeoDataFrame(columns=cols))
    
    return second_step
