# -*- coding: utf-8 -*-
import itertools
import logging
import os

import rasterio as rio
import rasterio.windows as riow

# ==================================================================================================
# Functions for creating tiles from raster files
# ==================================================================================================

log = logging.getLogger(__name__)


def get_tiles(
    geotiff: rio.DatasetReader,
    tile_width: int = 2000,
    tile_height: int = 2000,
    map_units: bool = False,
    offset: float = 0.0,
):

    """Defines a set of tiles over a raster layer based on user specified
    dimensions.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        tile_width (int, optional): Width of tile. Defaults to 2000.
        tile_height (int, optional): Height of tile. Defaults to 2000.
        map_units (bool, optional): If True, tile_width and tile_height are in map units. Defaults to False.
        offset (float, optional): Offset in percentage of tile. Defaults to 0.0.

    Yields:
        window (rio.windows.Window): Rasterio window object
        transform (Affine): Rasterio affine transform object
    """

    if map_units:
        if geotiff.transform.b == geotiff.transform.d == 0:
            # Get pixel size (x is width) (https://gis.stackexchange.com/questions/379005/using-raster-transform-function-of-rasterio-in-python)
            cell_x, _ = geotiff.transform.a, -geotiff.transform.e
            tile_width, tile_height = int(tile_width / cell_x + 0.5), int(
                tile_height / cell_x + 0.5
            )
        else:
            log.error("ValueError: Coefficient a from raster.transform.a is not width.")
            raise ValueError("Coefficient a from raster.transform.a is not width.")

    ncols, nrows = geotiff.meta["width"], geotiff.meta["height"]

    corners = itertools.product(
        range(0, ncols, tile_width), range(0, nrows, tile_height)
    )  # Not actual offsets, but just a grid of cells
    big_window = riow.Window(col_off=0, row_off=0, width=ncols, height=nrows)

    if offset > 0:
        log.error("NotImplementedError: Offset not implemented yet.")
        raise NotImplementedError("Offset not implemented yet.")
        offset = int(tile_width * offset)
        corners = itertools.product(
            range(offset, ncols, tile_width), range(offset, nrows, tile_height)
        )
        big_window = riow.Window(
            col_off=offset, row_off=offset, width=ncols - offset, height=nrows - offset
        )

    for col_corner, row_corner in corners:
        window = riow.Window(
            col_off=col_corner, row_off=row_corner, width=tile_width, height=tile_height
        ).intersection(big_window)
        transform = riow.transform(window, geotiff.transform)
        yield window, transform


def save_tiles(
    geotiff: rio.DatasetReader,
    out_path: str,
    tile_size: int = 2000,
    tile_template: str = "tile_{}-{}.tif",
):
    """Save tiles from a raster file.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        out_path (str): Path to save tiles to.
        tile_size (int): Size of tiles.
        tile_template (str): Template for tile names. Should contain two {} placeholders for the x and y coordinates of the tile.

    Returns:
        None
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        log.info(f"created '{out_path}'")
    else:
        log.info(f"'{out_path}' already exists")

    # with rio.open(raster_geotiffpath) as geotiff:
    tile_width, tile_height = tile_size, tile_size
    meta = geotiff.meta.copy()
    for window, transform in get_tiles(
        geotiff, tile_width, tile_height, map_units=True
    ):
        meta["transform"] = transform
        meta["width"], meta["height"] = window.width, window.height
        tile_path = os.path.join(
            out_path, tile_template.format(int(window.col_off), int(window.row_off))
        )
        with rio.open(tile_path, "w", **meta) as outds:
            outds.write(geotiff.read(window=window))
    # Close the big raster now that we are done with it.
    # geotiff.close()


def get_tiles_list_from_dir(tiles_dir: str):
    """Get a list of tiles from a directory.

    Args:
        tiles_dir (str): Path to tiles directory.

    Returns:
        tiles (list): List of rasterio raster objects.
    """

    tiles = []
    for tile in os.listdir(tiles_dir):
        tile_path = os.path.join(tiles_dir, tile)
        tiles.append(tile_path)
    return tiles


def load_tiles_from_list(tiles_list: list):
    """Load tiles from a list.

    Args:
        tiles_list (list): List of rasterio raster objects.

    Returns:
        tiles (list): List of rasterio raster objects.
    """

    tiles = []
    for tile in tiles_list:
        with rio.open(tile) as geotiff:
            tiles.append(geotiff)
    return tiles


def load_tiles_from_dir(tiles_dir: str):
    """Load tiles from a directory.

    Args:
        tiles_dir (str): Path to tiles directory.

    Returns:
        tiles (list): List of rasterio raster objects.
    """

    tiles = []
    for tile in os.listdir(tiles_dir):
        tile_path = os.path.join(tiles_dir, tile)
        with rio.open(tile_path) as geotiff:
            tiles.append(geotiff)
    return tiles
