import rasterio as rio
import itertools
import rasterio.windows as riow
import os

# ==================================================================================================
# Functions for creating tiles from raster files
# ==================================================================================================

def get_tiles(geotiff:rio.DatasetReader, tile_width:int=2000, tile_height:int=2000, map_units:bool=False, offset:float=0.0):

    """
    Defines a set of tiles over a raster layer based on user specified dimensions.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        tile_width (int, optional): Width of tile in pixels. Defaults to 2000.
        tile_height (int, optional): Height of tile in pixels. Defaults to 2000.
        map_units (bool, optional): If True, tile_width and tile_height are in map units. Defaults to False.
        offset (float, optional): Offset in percentage of tile. Defaults to 0.0.
    
    Yields:
        window (rio.windows.Window): Rasterio window object
        transform (Affine): Rasterio affine transform object
    """

    if map_units:
        if geotiff.transform.b==geotiff.transform.d==0:
        # Get pixel size (x is width) (https://gis.stackexchange.com/questions/379005/using-raster-transform-function-of-rasterio-in-python)
            cell_x, _ = geotiff.transform.a, -geotiff.transform.e 
            tile_width, tile_height = int(tile_width / cell_x + 0.5) , int(tile_height / cell_x + 0.5)
        else:
            raise ValueError("Coefficient a from raster.transform.a is not width.")

    ncols, nrows = geotiff.meta['width'], geotiff.meta['height']

    corners = itertools.product(range(0, ncols, tile_width), range(0, nrows, tile_height)) # Not actual offsets, but just a grid of cells
    big_window = riow.Window(col_off=0, row_off=0, width=ncols, height=nrows)

    if offset>0:
        offset = int(tile_width*offset)
        corners = itertools.product(range(offset, ncols, tile_width), range(offset, nrows, tile_height))
        big_window = riow.Window(col_off=offset, row_off=offset, width=ncols-offset, height=nrows-offset)

    
    for col_corner, row_corner in  corners:
        window = riow.Window(col_off=col_corner, row_off=row_corner, width=tile_width, height=tile_height).intersection(big_window)
        transform = riow.transform(window, geotiff.transform)
        yield window, transform


def save_tiles(geotiff:rio.DatasetReader, out_path:str, tile_size:int=2000, tile_template:str = "tile_{}-{}.tif"):
    """
    Save tiles from a raster file.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        out_path (str): Path to save tiles to.
        tile_size (int): Size of tiles in pixels.
        tile_template (str): Template for tile names. Should contain two {} placeholders for the x and y coordinates of the tile.

    Returns:
        None
    """
        
    # with rio.open(raster_geotiffpath) as geotiff:
    tile_width, tile_height = tile_size, tile_size 
    meta = geotiff.meta.copy()
    for window, transform in get_tiles(geotiff, tile_width, tile_height, map_units=True):
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        outpath = os.path.join(out_path,tile_template.format(int(window.col_off), int(window.row_off)))
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(geotiff.read(window=window))
    # Close the big raster now that we are done with it.
    # geotiff.close()