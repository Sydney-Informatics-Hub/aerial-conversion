# -*- coding: utf-8 -*-
import argparse

# import glob
import logging

# import os
# import os.path
# import traceback
from pathlib import Path

from libs.tiles import get_tiles_list_from_dir, load_tiles_from_list

# import geopandas as gpd
# import rasterio as rio


log = logging.getLogger(__name__)


#%% Command-line driver


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tile-dir",
        required=True,
        type=Path,
        help="Path to the input tiles directory.",
    )
    ap.add_argument(
        "--json-name",
        default="coco_from_gis.json",
        type=Path,
        help="Path to the input coco json file.",
    )
    ap.add_argument(
        "--crs", type=str, default=None, help="Specifiy the project crs to use."
    )
    ap.add_argument("--tile-extension", default="tif", type=str)
    ap.add_argument(
        "--polygon-file",
        required=True,
        default=".",
        type=Path,
        help="Path to output polygon file.",
    )
    ap.add_argument(
        "--raster-file", required=True, type=Path, help="Path to output raster file."
    )
    ap.add_argument(
        "--license",
        type=Path,
        help="Path to a license description in COCO JSON format. If not supplied, will default to MIT license.",
    )
    args = ap.parse_args(args)

    """
    Create tiles from raster and convert to COCO JSON format.
    """
    # root_dir = "/home/sahand/Data/GIS2COCO/"
    # raster_path = os.path.join(root_dir, "chatswood/chatswood_hd.tif")
    # geojson_path = os.path.join(root_dir, "chatswood/chatswood.geojson")
    # tile_dir = os.path.join(root_dir, "chatswood/tiles/")
    # user_crs = None
    # license = None
    # json_name = os.path.join(root_dir, "chatswood/coco_from_gis.json")

    # raster_path = args.raster_file
    # geojson_path = args.polygon_file
    tile_dir = args.tile_dir
    # user_crs = args.crs
    # license = args.license
    # json_name = args.json_name
    tile_extension = args.tile_extension

    # Read tiles
    log.info("Reading tiles from %s" % tile_dir)
    tiles_list = get_tiles_list_from_dir(tiles_dir=tile_dir, extension=tile_extension)
    rasters = load_tiles_from_list(tiles_list=tiles_list)

    print(len(rasters))


if __name__ == "__main__":
    main()
