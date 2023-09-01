import os 
import itertools
import rasterio as rio
import pandas as pd
import geopandas as gpd
from rasterio import windows as riow
from shapely.geometry import MultiPoint
import fiona

# ==================================================================================================
# Functions for converting between spatial and pixel coordinates
# ==================================================================================================

def wkt_parser(wkt_str:str):
    """
    Parses a WKT string to extract the local coordinate system.

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
            return x
        if x == "LOCAL_CS[":
            set = True
    return wkt_str


def reproject_coords(src_crs, dst_crs, coords):
    """
    Reprojects a list of coordinates from one coordinate system to another.

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
    return [[x,y] for x,y in zip(xs, ys)]


def pixel_to_spatial_rio(raster,x,y):
    """ 
    Converts pixel coordinates to spatial coordinates using rasterio.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        x (int): x coordinate in pixels
        y (int): y coordinate in pixels

    Returns:
        tuple: (x,y) spatial coordinates
    
    """

    return raster.xy(x,y)  #px, py
    

def spatial_to_pixel_rio(raster, x, y):
    """
    Converts spatial coordinates to pixel coordinates using rasterio.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        x (float): longitudinal coordinate in spatial units
        y (float): latitudinal coordinate in spatial units

    Returns:
        tuple: (x,y) pixel coordinates

    """

    px,py = raster.index(x,y) # lon,lat
    return px,py


def spatial_polygon_to_pixel_rio(raster, polygon):    
    """
    Converts spatial polygon to pixel polygon using rasterio.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        polygon (shapely.geometry.Polygon): Polygon in spatial coordinates

    Returns:
        converted_coords (list): List of pixel coordinates defining the polygon

    """
    converted_coords = []
    for point in list(MultiPoint(polygon.exterior.coords).geoms):
        x, y = spatial_to_pixel_rio(raster, point.x, point.y)
        pixel_point = x, y
        converted_coords.append(pixel_point)
    return(converted_coords)






   

