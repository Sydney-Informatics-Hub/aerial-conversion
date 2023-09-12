# -*- coding: utf-8 -*-
"""A collection of functions for converting between spatial and pixel
coordinates.

This module contains functions for converting between spatial and pixel coordinates, as well as functions for creating
pixel polygons for a list of raster tiles. The pixel polygons are used to create COCO annotations for each polygon.

Also contains functions for creating polygons from coco annotations.
"""

import logging

import fiona
import geopandas as gpd
import pandas as pd
import rasterio as rio
from shapely.geometry import MultiPoint, box
from tqdm import tqdm

log = logging.getLogger(__name__)


def wkt_parser(wkt_str: str):
    """Parses a WKT string to extract the local coordinate system.

    Args:
        wkt_str (str): WKT string

    Returns:
        str: Local coordinate system
    """
    # TODO: Make this tool smarter. Right now it just looks for the first LOCAL_CS[ and returns everything after that.

    wkt = wkt_str.split('"')
    set = False
    for x in wkt:
        if set is True:
            log.debug(f"LOCAL_CS is {x}")
            return x
        if x == "LOCAL_CS[":
            log.debug(f"Found LOCAL_CS[ at {wkt.index(x)}")
            set = True
    log.info(f"wtkt_str: {wkt_str}")
    return wkt_str


def reproject_coords(src_crs, dst_crs, coords):
    """Reprojects a list of coordinates from one coordinate system to another.

    Args:
        src_crs (str): Source coordinate system
        dst_crs (str): Destination coordinate system
        coords (list): List of coordinates to reproject

    Returns:
        list: List of reprojected coordinates
    """

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xs, ys = fiona.transform.transform(src_crs, dst_crs, xs, ys)
    return [[x, y] for x, y in zip(xs, ys)]


def pixel_to_spatial_rio(raster, row_ind, col_ind):
    """Converts pixel coordinates to spatial coordinates using rasterio.
        More information here: https://stackoverflow.com/questions/52443906/pixel-array-position-to-lat-long-gdal-python
    Args:
        raster (rio.DatasetReader): Rasterio raster object
        row_ind (int): pixel row
        col_ind (int): pixel column

    Returns:
        tuple: (x,y) spatial coordinates
    """

    return raster.xy(row_ind, col_ind)  # px, py


def spatial_to_pixel_rio(raster, x, y):
    """Converts spatial coordinates to pixel coordinates using rasterio.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        x (float): longitudinal coordinate in spatial units
        y (float): latitudinal coordinate in spatial units

    Returns:
        tuple: (row_ind,col_ind) pixel coordinates
    """

    row_ind, col_ind = raster.index(x, y)  # lon,lat
    return row_ind, col_ind


def spatial_polygon_to_pixel_rio(raster, polygon) -> list:
    """Converts spatial polygon to pixel polygon using rasterio.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        polygon (shapely.geometry.Polygon): Polygon in spatial coordinates

    Returns:
        converted_coords (list): List of pixel coordinates defining the polygon
    """
    converted_coords = []
    for point in list(MultiPoint(polygon.exterior.coords).geoms):
        log.debug(f"Converting {point} to pixel coordinates in raster {raster}")
        y, x = spatial_to_pixel_rio(raster, point.x, point.y)
        pixel_point = x, y
        converted_coords.append(pixel_point)
    return converted_coords


def get_tile_polygons(raster_tile: str, geojson: gpd.GeoDataFrame, filter: int = 0):
    """Create polygons from a geosjon for an individual raster tile.

    Args:
        raster_tile: (str) a file name referring to the raster tile to be loaded
        geojson: (gpd.GeoDataFrame) a geodataframe with polygons
        filter: (int) an integer to filter out polygons with area less than the filter value

    Returns:
        tile_polygon: geodataframe with polygons within the raster's extent
    """
    # print(geojson.shape)
    # Load raster tile
    raster_tile = rio.open(raster_tile)
    raster_extent = gpd.GeoDataFrame(
        {"id": 1, "geometry": [box(*raster_tile.bounds)]}, crs=geojson.crs
    )
    # geojson = geojson.to_crs(geojson)
    tile_polygons = geojson.clip(raster_extent)

    # Split multipolygon
    tile_polygons = tile_polygons.explode(index_parts=False)
    tile_polygons = tile_polygons.reset_index(drop=True)
    # Filter out zero area polygons
    tile_polygons = tile_polygons[tile_polygons.geometry.area > filter]
    # if filter is True:
    #     tile_polygons = tile_polygons[tile_polygons.geometry.area > 5000]
    tile_polygons = tile_polygons.reset_index(drop=True)
    # print(tile_polygons.shape)
    return tile_polygons


def pixel_polygons_for_raster_tiles(
    raster_file_list: list, geojson: gpd.GeoDataFrame, verbose=1
):
    """Create pixel polygons for a list of raster tiles.

    Args:
        raster_file_list (list): List of raster files
        geojson (gpd.GeoDataFrame): GeoDataFrame containing polygons
        verbose (int): Verbosity level

    Returns:
        pixel_df (pd.DataFrame): DataFrame containing pixel polygons
    """
    tmp_list = []
    log.info(f"Creating pixel polygons for {len(raster_file_list)} tiles")
    for index, file in enumerate(raster_file_list):
        tmp = get_tile_polygons(file, geojson, 0)
        tmp["raster_tile"] = rio.open(file)
        tmp["image_id"] = index
        tmp_list.append(tmp)

    log.info(f"Concatenating {len(tmp_list)} GeoDataFrames")
    pixel_df = pd.concat(tmp_list).reset_index()
    pixel_df = pixel_df.drop(columns=["index"])
    log.info(f"Creating pixel polygons for {pixel_df.shape[0]} polygons")
    if verbose > 0:
        tqdm.pandas()
        pixel_df["pixel_polygon"] = pixel_df.progress_apply(
            lambda row: spatial_polygon_to_pixel_rio(
                row["raster_tile"], row["geometry"]
            ),
            axis=1,
        )
    else:
        pixel_df["pixel_polygon"] = pixel_df.apply(
            lambda row: spatial_polygon_to_pixel_rio(
                row["raster_tile"], row["geometry"]
            ),
            axis=1,
        )
    pixel_df["annot_id"] = range(0, 0 + len(pixel_df))
    log.info(f"Pixel polygons created for {pixel_df.shape[0]} polygons")
    return pixel_df
