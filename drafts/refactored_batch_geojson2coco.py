# -*- coding: utf-8 -*-
"""This script supports batch conversion of paired geojson and raster data into
a series of COCO datasets."""
import argparse
import json
import logging
import os
import pickle
import subprocess

import geopandas as gpd
import pandas as pd
import rasterio
from pycocotools.coco import COCO
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm

from aerial_conversion.coordinates import wkt_parser

# set up logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def resume(output_dir: str) -> list:
    """Resume a batch job from an output directory.

    Check for the directories that are already processed, and check if they have a .json in them, then record them as already processed.

    Args:
        output_dir (str): Path to the output directory.

    Returns:
        list: List of raster files that are already processed.
    """

    processed = []
    for sub_dir in os.listdir(output_dir):
        # Check for all subdirs
        if os.path.isdir(os.path.join(output_dir, sub_dir)):
            # Check if they have a .json file in them
            if os.path.exists(os.path.join(output_dir, sub_dir, "coco_from_gis.json")):
                processed.append(sub_dir)

    return processed

def crop_and_save_geojson(raster_dir: str, geojson_path: str, raster_extension: str = ".tif", user_crs=None):
    """Crop a GeoJSON file to the extent of a raster file and save it.

    Args:
        raster_dir (str): Path to the directory containing the raster files.
        geojson_path (str): Path to the GeoJSON file.
        raster_extension (str, optional): Extension of the raster files. Defaults to '.tif'.
        user_crs ([type], optional): CRS of the raster files. Defaults to None.
    """

    # Read the GeoJSON file
    geojson = gpd.read_file(geojson_path)

    cropped_dir = os.path.join(
        os.path.dirname(geojson_path), os.path.basename(geojson_path).split(".")[0]
    )

    # Create the cropped directory
    os.makedirs(cropped_dir, exist_ok=True)

    # Loop through each raster file
    for raster_file in tqdm(os.listdir(raster_dir)):
        if raster_file.endswith(raster_extension):
            raster_path = os.path.join(raster_dir, raster_file)

            # Open the raster file and get its bounds
            with rasterio.open(raster_path) as src:
                left, bottom, right, top = src.bounds

            # Find the crs of the raster
            if user_crs is None:
                user_crs = src.crs.to_wkt()
                user_crs = wkt_parser(user_crs)

            # Create a bounding box from the raster bounds
            bbox = box(left, bottom, right, top)

            # Reproject the GeoJSON to the CRS of the raster
            geojson_crs = geojson.crs
            if geojson_crs != user_crs:
                geojson = geojson.to_crs(user_crs)

            # Crop the GeoJSON to the extent of the raster
            cropped_geojson = geojson[geojson.geometry.intersects(bbox)]

            # Save the cropped GeoJSON with the same naming pattern
            cropped_geojson_filename = os.path.join(
                cropped_dir, os.path.basename(raster_file).split(".")[0] + ".geojson"
            )
            cropped_geojson.to_file(cropped_geojson_filename, driver="GeoJSON")

    return cropped_dir

def process_single(args):
    output_dir = args.output_dir

    individual_coco_datasets = []  # List to store individual COCO datasets
    error = {}
    if args.resume:
        already_processed = resume(output_dir)
        print(f"Resuming from {len(already_processed)} already processed files.")
    else:
        already_processed = []

    # Iterate over the raster directory
    for raster_file in os.listdir(args.raster_dir):
        # Check if the file is a GeoTIFF
        if raster_file.endswith(".tif"):
            # Get the file name without extension
            file_name = os.path.basename(raster_file).split(".")[0]
            if file_name in already_processed:
                print(f"Skipping {file_name} as it is already processed.")
                # add the json file to the list
                individual_coco_datasets.append(
                    os.path.join(output_dir, file_name, "coco_from_gis.json")
                )
                continue
            # Construct the vector file name
            vector_file = file_name + args.pattern + ".geojson"

            # print(f"Processing {vector_file} | {raster_file}")

            vector_path = os.path.join(args.vector_dir, vector_file)

            # Check if the vector file exists
            if os.path.exists(vector_path):
                # Specify the output directory for the file pair
                pair_output_dir = os.path.join(output_dir, file_name)
                os.makedirs(pair_output_dir, exist_ok=True)

                # Specify the output JSON file path
                json_file = os.path.join(pair_output_dir, "coco_from_gis.json")

                if not args.info:
                    # create an empty info.json file and set info_file to it
                    info_file = os.path.join(output_dir, "info.json")
                    with open(info_file, "w") as f:
                        json.dump({}, f)
                else:
                    info_file = os.path.join(args.info, f"{file_name}.json")

                # Specify the overlap
                overlap = args.overlap

                print(
                    f"Processing {vector_file} | {raster_file} | {info_file} > > > > {json_file}"
                )

                # Construct the command
                command = [
                    "geojson2coco",
                    "--raster-file",
                    os.path.join(args.raster_dir, raster_file),
                    "--polygon-file",
                    vector_path,
                    "--tile-dir",
                    pair_output_dir,
                    "--json-name",
                    json_file,
                    "--offset",
                    str(overlap),
                    "--info",
                    info_file,
                    "--tile-size",
                    str(args.tile_size),
                    "--class-column",
                    args.class_column,
                ]
                try:
                    # Run the command
                    subprocess.run(command, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {vector_file}: {e.stderr}")
                    error[file_name] = e.stderr

                    # Save the error as csv as well using pandas pd
                    df_errors = pd.DataFrame.from_dict(
                        {file_name: e.stderr}, orient="index"
                    )
                    df_errors.columns = ["error_message"]
                    df_errors.to_csv(os.path.join(output_dir, "error.csv"), mode="a")

                # Add the generated COCO dataset to the list
                individual_coco_datasets.append(json_file)

    # Save the error dict
    with open(os.path.join(output_dir, "error.pkl"), "wb") as f:
        pickle.dump(error, f)

    return individual_coco_datasets

def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Convert raster and vector pairs to COCO JSON format."
    )
    parser.add_argument(
        "--raster-dir", required=True, help="Path to the raster directory."
    )
    parser.add_argument(
        "--vector-dir", required=True, help="Path to the vector directory."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "--tile-size", type=float, default=100, help="Tile width/height in meters."
    )
    parser.add_argument(
        "--class-column",
        required=True,
        help="Column name in GeoJSON for classes. Should always be provided. If the column does not exist, will create a new column with the name provided.",
    )
    parser.add_argument(
        "--overlap",
        default=0,
        help="Overlap between tiles in percentage. Defaults to 0.",
    )
    parser.add_argument(
        "--pattern",
        default="",
        help="Pattern to match the vector file name. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--concatenate",
        action=argparse.BooleanOptionalAction,
        help="Concatenate individual COCO datasets into one.",
    )
    parser.add_argument(
        "--info",
        help="Path to the info JSON file. Either leave empty, or put a folder path., containing info.json for each raster, with the same names as the rasters.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        help="Resume a batch job from an output directory.",
    )
    parser.add_argument(
        "--no-workers",
        type=int,
        default=1,
        help="Number of workers to use for parallel processing.",
    )
    parser.add_argument(
        "--user_assumed_raster_crs",
        type=str,
        default=None,
        help="If the raster crs is not defined in the raster file, you can provide it here. It will be used to crop the geojson file to the extent of the raster file.",
    )

    return parser.parse_args(args)

def process_vector_dir(args):
    """
    Check and process the vector directory.

    Args:
    - args: Command-line arguments.

    This function checks if the provided vector directory exists. If the directory is not found and the provided file is a GeoJSON file, it crops the GeoJSON file to the extent of the raster file specified.

    Returns:
    - args: Updated command-line arguments.
    """
    if not os.path.isdir(args.vector_dir):
        if args.vector_dir.endswith(".geojson"):
            logger.info(
                "The vector-dir is not a directory, and is a geojson file. Cropping it to the extent of the raster file."
            )
            args.vector_dir = crop_and_save_geojson(
                args.raster_dir,
                args.vector_dir,
                raster_extension=".tif",
                user_crs=args.user_assumed_raster_crs,
            )
        else:
            raise ValueError(
                "The vector-dir is not a directory, and is not a geojson file. Please provide a directory or a geojson file."
            )

    return args

def print_individual_coco_datasets(individual_coco_datasets):
    """Print markdown output for individual COCO datasets."""
    print("Running geojson2coco.py over raster and vector pairs:")
    print()
    print("| Raster File\t| Vector File\t| JSON File\t|")
    print("|------------\t|------------\t|----------\t|")
    for coco_file in individual_coco_datasets:
        pair_dir = os.path.dirname(coco_file)
        raster_file = os.path.basename(pair_dir) + ".tif"
        vector_file = os.path.basename(pair_dir) + ".geojson"
        print(f"| {raster_file}\t| {vector_file}\t| {coco_file}\t|")

def concatenate_datasets(individual_coco_datasets: list[Path], output_dir: Path) -> None:
    """
    Concatenate individual COCO datasets into a single dataset.

    Args:
        individual_coco_datasets (List[Path]): Paths to individual COCO dataset JSON files.
        output_dir (Path): Directory where the concatenated output should be saved.
    """
    concatenated_coco = {
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
        "info": {},
    }

    category_index_checkpoint = 0
    image_index_checkpoint = 0
    annot_index_checkpoint = 0

    category_name_to_id = {}

    for coco_file in tqdm(individual_coco_datasets):
        try:
            with open(coco_file, "r") as f:
                dataset = json.load(f)
        except FileNotFoundError:
            print(f"Error: {coco_file} not found.")
            continue

        raster_name = os.path.basename(os.path.dirname(coco_file))

        # Remap image IDs and file names
        image_id_map = {}
        for image in dataset["images"]:
            image_id_map[image["id"]] = image_index_checkpoint
            image["id"] = image_index_checkpoint
            image["file_name"] = os.path.join(raster_name, image["file_name"])
            concatenated_coco["images"].append(image)
            image_index_checkpoint += 1

        # Handle categories and annotations
        for category in dataset["categories"]:
            category_id = category_name_to_id.setdefault(category["name"], category_index_checkpoint)
            if category_id == category_index_checkpoint:
                category["id"] = category_index_checkpoint
                concatenated_coco["categories"].append(category)
                category_index_checkpoint += 1
            
        for annotation in dataset["annotations"]:
            annotation["image_id"] = image_id_map[annotation["image_id"]]
            annotation["id"] = annot_index_checkpoint
            annotation["category_id"] = category_name_to_id[annotation["category_id"]]
            if not isinstance(annotation["segmentation"][0], list):
                annotation["segmentation"] = [annotation["segmentation"]]
            concatenated_coco["annotations"].append(annotation)
            annot_index_checkpoint += 1

        # Extend licenses and info if present
        concatenated_coco["licenses"].extend(dataset.get("licenses", []))
        concatenated_coco["info"].update(dataset.get("info", {}))

    # Specify and create the output directory for the concatenated dataset
    concatenated_output_dir = output_dir / "concatenated"
    concatenated_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the concatenated COCO dataset
    concatenated_json_file = concatenated_output_dir / "concatenated_coco.json"
    with open(concatenated_json_file, "w") as f:
        json.dump(concatenated_coco, f, indent=2)

    print(f"\nConcatenated COCO dataset saved to: {concatenated_json_file}")


def main(args=None):
    """Convert raster and vector pairs to COCO JSON format.

    Args:
        args: Command-line arguments.

    Usage Example:
        # Convert raster and vector pairs without concatenation
        python batch_geojson2coco.py --raster-dir /path/to/raster_dir --vector-dir /path/to/vector_dir --output-dir /path/to/output_dir

        # Convert raster and vector pairs with concatenation
        python batch_geojson2coco.py --raster-dir /path/to/raster_dir --vector-dir /path/to/vector_dir --output-dir /path/to/output_dir --concatenate
    """

    args = parse_arguments(args)
    
    # Check the vector-dir, and if it is not a dir, and is a single geojson file, then crop it to the extent of the raster file
    args = process_vector_dir(args)

        # Specify the output directory
    if args.no_workers > 1:
        raise NotImplementedError("Parallel processing not implemented yet.")
    else:
        print("Running geojson2coco.py over raster and vector pairs:")
        individual_coco_datasets = process_single(args)
        # Generate markdown output for individual COCO datasets
        print_individual_coco_datasets(individual_coco_datasets)

    # Concatenate COCO datasets if the --concatenate argument is enabled
    if args.concatenate:
        concatenate_datasets(individual_coco_datasets, Path(args.output_dir))

if __name__ == "__main__":
    main()
