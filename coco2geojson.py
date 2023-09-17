# -*- coding: utf-8 -*-
import argparse

# import glob
import logging
import os
import warnings

# import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from libs.coco import coco_annotation_per_image_df, coco_categories_dict
from libs.coordinates import pixel_segmentation_to_spatial_rio, read_crs_from_raster
from libs.tiles import get_tiles_list_from_dir, load_tiles_from_list

# import rasterio as rio
warnings.simplefilter(action="ignore", category=FutureWarning)


log = logging.getLogger(__name__)


def assemble_geo_json(
    crs,
    meta_name,
    meta_type,
    properties_json,
    coordinates_z,
    license_json,
    info_json,
    categories_json,
):
    pass


#%% Command-line driver


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tile-dir",
        # required=True,
        type=Path,
        default="/home/sahand/Data/GIS2COCO/chatswood/big_tiles_200/",
        help="Path to the input tiles directory.",
    )
    ap.add_argument(
        "--coco-json",
        default="/home/sahand/Data/GIS2COCO/chatswood/big_tiles_200/coco_from_gis_hd_200.json",
        type=Path,
        help="Path to the input coco json file.",
    )
    ap.add_argument(
        "--tile-extension",
        default="tif",
        type=str,
        help="Extension of tiles. Defaults to 'tif'.",
    )
    ap.add_argument(
        "--geojson-output",
        # required=True,
        default="/home/sahand/Data/GIS2COCO/chatswood/coco_2_geojson.geojson",
        type=Path,
        help="Path to output geojson file.",
    )
    ap.add_argument(
        "--tile-search-margin",
        default=5,
        type=int,
        help="Int percentage of tile size to use as a search margin for finding overlapping polygons while joining raster. Defaults to 10%.",
    )
    ap.add_argument(
        "--not-keep-geom-type",
        action=argparse.BooleanOptionalAction,
        help="If not set, return only geometries of the same geometry type as df1 has, otherwise, return all resulting geometries. It it advised not to set this parameter. It is known to casue polygon matching issues.",
    )
    ap.add_argument(
        "--meta-name",
        type=str,
        default="Aerial Segmentation Predictions",
        help="Name of the prediction.",
    )
    ap.add_argument(
        "--meta-type",
        type=str,
        default="FeatureCollection",
        help="Type of the output geojson file.",
    )
    ap.add_argument(
        "--properties-json",
        type=Path,
        default=None,
        help="Path to a JSON file containing common properties to add to the geojson file for each annotation.",
    )
    ap.add_argument(
        "--coordinates-z",
        type=float,
        default=None,
        help="A common Z coordinate to use for all annotations. If not set, will not add Z coordinates.",
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
    geojson_path = args.geojson_output
    tile_dir = args.tile_dir
    meta_name = args.meta_name
    coco_json_path = args.coco_json
    tile_extension = args.tile_extension
    keep_geom_type = not args.not_keep_geom_type  # should be True
    tile_search_margin = args.tile_search_margin

    # Read tiles
    log.info("Reading tiles from %s" % tile_dir)
    tiles_list = get_tiles_list_from_dir(tile_dir, tile_extension)
    geotifs = load_tiles_from_list(tiles_list=tiles_list)
    crs = read_crs_from_raster(tiles_list[0])

    # Read COCO JSON and extract annotations per reference image
    tiles_df = pd.DataFrame(
        zip(tiles_list, geotifs), columns=["raster_path", "geotiff"]
    )
    tiles_df["tile_name"] = tiles_df["raster_path"].apply(
        lambda x: os.path.basename(x).split(".")[0]
    )

    coco_images_df = coco_annotation_per_image_df(coco_json_path, tile_search_margin)
    coco_categories = coco_categories_dict(coco_json_path)
    coco_images_df["annotations"][0]["bbox"]

    # Merge COCO JSON with tiles
    tiles_df = pd.merge(tiles_df, coco_images_df, on="tile_name", how="outer")

    # Unpack annotations to segmentation column, yielding longer dataframe
    tiles_df.dropna(subset=["annotations"], inplace=True)
    tiles_df["segmentation"] = tiles_df["annotations"].apply(
        lambda x: x["segmentation"]
    )
    tiles_df["bbox"] = tiles_df["annotations"].apply(lambda x: x["bbox"])
    tiles_df["zone_code"] = tiles_df["annotations"].apply(lambda x: x["category_id"])
    tiles_df["zone_name"] = tiles_df["zone_code"].apply(
        lambda x: coco_categories[x]["name"]
    )
    tiles_df["marginal"] = tiles_df["annotations"].apply(lambda x: x["marginal"])
    tiles_df = tiles_df.drop(columns=["annotations"])

    # tiles_df[tiles_df["marginal"]==False]

    # Group by zone ID, extract polygons, and merge overlapping polygons in the same zone
    tiles_df_grouped = tiles_df.groupby(["zone_code"]).groups
    tiles_df_zone_groups = list()
    for zone in tiles_df_grouped:
        tiles_df_zone_groups.append(tiles_df.loc[tiles_df_grouped[zone]])

    # Create a list of GeoDataFrames, one per zone, after convering the pixel coordinates to raster coordinates
    polygons_df_zone_groups = list()
    for tiles_df_zone in tiles_df_zone_groups:
        tiles_df_zone = tiles_df_zone.reset_index(drop=True)

        # Convert segmentations to polygons
        tiles_df_zone["geometry"] = tiles_df_zone.apply(
            lambda x: pixel_segmentation_to_spatial_rio(
                x["geotiff"], x["segmentation"]
            ),
            axis=1,
        )

        # Create a GeoDataFrame with the polygons
        for i, row in tqdm(tiles_df_zone.iterrows(), total=tiles_df_zone.shape[0]):
            polygons_df_tmp = gpd.GeoDataFrame(crs=crs, geometry=[row["geometry"]])
            # polygons_df_tmp["zone_code"] = row["zone_code"]
            # polygons_df_tmp["zone_name"] = row["zone_name"]
            # polygons_df_tmp["tile"] = row["tile_name"]
            if not polygons_df_tmp.empty:
                if i == 0:
                    polygons_df_zone = polygons_df_tmp.copy()
                else:
                    # Merge the GeoDataFrames and combine overlapping polygons
                    if row["marginal"] is True:
                        polygons_df_zone = gpd.overlay(
                            polygons_df_zone,
                            polygons_df_tmp,
                            how="union",
                            keep_geom_type=keep_geom_type,
                        )  # .reset_index(drop=True)
                        # TODO: Fix the following warning, or be careful abou the versions. (pandas==2.1.0, geopandas==0.13.2):
                        # FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
                        #   In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes.
                        #   To retain the old behavior, exclude the relevant entries before the concat operation
                    else:
                        polygons_df_zone = pd.concat(
                            [polygons_df_zone, polygons_df_tmp], ignore_index=True
                        )

            # print(polygons_df_zone)
        polygons_df_zone["zone_code"] = row["zone_code"]
        polygons_df_zone["zone_name"] = row["zone_name"]
        polygons_df_zone_groups.append(polygons_df_zone)

    polygons_df = pd.concat(polygons_df_zone_groups, ignore_index=True)
    try:
        polygons_df.Name = meta_name
    except Exception as e:
        log.error(f"Could not set Name property of geojson. Error message: {e}")
        print("FIX this code!")
    # Save to geojson
    polygons_df.to_file(geojson_path, driver="GeoJSON")


if __name__ == "__main__":
    main()
