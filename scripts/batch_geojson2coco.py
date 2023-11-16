# -*- coding: utf-8 -*-
"""This script supports batch conversion of paired geojson and raster data into
a series of COCO datasets."""
import argparse
import json
import os
import pickle
import subprocess

import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm

# from subprocess import Popen


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
            file_name = os.path.splitext(raster_file)[0].split("_")[0]
            if file_name in already_processed:
                print(f"Skipping {file_name} as it is already processed.")
                # add the json file to the list
                individual_coco_datasets.append(
                    os.path.join(output_dir, file_name, "coco_from_gis.json")
                )
                continue
            # Construct the vector file name
            vector_file = file_name + args.pattern + ".geojson"
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
                    "python",
                    "scripts/geojson2coco.py",
                    "--raster-file",
                    os.path.join(args.raster_dir, raster_file),
                    "--polygon-file",
                    vector_path,
                    "--tile-dir",
                    pair_output_dir,
                    "--json-name",
                    json_file,
                    "--offset",
                    overlap,
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


def main(args):
    """Convert raster and vector pairs to COCO JSON format.

    Args:
        args: Command-line arguments.

    Usage Example:
        # Convert raster and vector pairs without concatenation
        python batch_geojson2coco.py --raster-dir /path/to/raster_dir --vector-dir /path/to/vector_dir --output-dir /path/to/output_dir

        # Convert raster and vector pairs with concatenation
        python batch_geojson2coco.py --raster-dir /path/to/raster_dir --vector-dir /path/to/vector_dir --output-dir /path/to/output_dir --concatenate
    """

    # Specify the output directory
    if args.no_workers > 1:
        raise NotImplementedError("Parallel processing not implemented yet.")
    else:
        individual_coco_datasets = process_single(args)

    # Generate markdown output for individual COCO datasets
    print("Running geojson2coco.py over raster and vector pairs:")
    print()
    print("| Raster File | Vector File | JSON File |")
    print("|-------------|-------------|-----------|")
    for coco_file in individual_coco_datasets:
        pair_dir = os.path.dirname(coco_file)
        raster_file = os.path.basename(pair_dir) + ".tif"
        vector_file = os.path.basename(pair_dir) + ".geojson"
        print(f"| {raster_file} | {vector_file} | {coco_file} |")

    # Concatenate COCO datasets if the --concatenate argument is enabled
    if args.concatenate:
        concatenated_coco = COCO()  # Create a new COCO dataset
        concatenated_coco.dataset = {
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [],
            "info": {},
        }

        # Fix the category ids in annotations and categories blocks
        category_index_checkpoint = 0
        image_index_checkpoint = 0
        annot_index_checkpoint = 0
        for coco_file in tqdm(individual_coco_datasets):
            image_index_map = {}
            category_index_map = {}

            try:
                with open(coco_file, "r") as f:
                    dataset = json.load(f)
            except FileNotFoundError:
                print(f"Error: {coco_file} not found.")
                continue

            pair_dir = os.path.dirname(coco_file)
            raster_name = os.path.basename(pair_dir)

            for image_no, _ in enumerate(dataset["images"]):
                dataset["images"][image_no]["file_name"] = os.path.join(
                    raster_name, dataset["images"][image_no]["file_name"]
                )

                image_index_map[
                    dataset["images"][image_no]["id"]
                ] = image_index_checkpoint

                dataset["images"][image_no]["id"] = image_index_checkpoint
                image_index_checkpoint += 1

            for _, dataset_category in enumerate(dataset["categories"]):
                old_id = dataset_category["id"]

                if dataset_category["name"] not in [
                    category["name"]
                    for category in concatenated_coco.dataset["categories"]
                ]:
                    dataset_category["id"] = category_index_checkpoint
                    concatenated_coco.dataset["categories"].append(dataset_category)
                    category_index_map[old_id] = category_index_checkpoint
                    category_index_checkpoint += 1

                else:
                    # find the existing mapping id
                    existing_mapping_id = None
                    for category in concatenated_coco.dataset["categories"]:
                        if category["name"] == dataset_category["name"]:
                            existing_mapping_id = category["id"]
                            break
                    dataset_category["id"] = existing_mapping_id
                    category_index_map[old_id] = existing_mapping_id

            for annotation_no, _ in enumerate(dataset["annotations"]):
                annotation_image_id = dataset["annotations"][annotation_no]["image_id"]
                dataset["annotations"][annotation_no]["image_id"] = image_index_map[
                    annotation_image_id
                ]
                dataset["annotations"][annotation_no]["id"] = annot_index_checkpoint

                # make the segnmets list of lists if not already
                if not isinstance(
                    dataset["annotations"][annotation_no]["segmentation"][0], list
                ):
                    dataset["annotations"][annotation_no]["segmentation"] = [
                        dataset["annotations"][annotation_no]["segmentation"]
                    ]

                # fix the annotation category id by the category_index_map
                dataset["annotations"][annotation_no][
                    "category_id"
                ] = category_index_map[
                    dataset["annotations"][annotation_no]["category_id"]
                ]

                annot_index_checkpoint += 1

            # Add the dataset to the concatenated COCO dataset
            concatenated_coco.dataset["images"].extend(dataset["images"])
            concatenated_coco.dataset["annotations"].extend(dataset["annotations"])

            # Add the categories to the concatenated COCO dataset if dataset["categories"]["id"] are not already in the concatenated_coco.dataset["categories"]["id"]
            for category in dataset["categories"]:
                if category["id"] not in [
                    category["id"]
                    for category in concatenated_coco.dataset["categories"]
                ]:
                    concatenated_coco.dataset["categories"].append(category)
            try:
                concatenated_coco.dataset["licenses"].extend(dataset["licenses"])
            except KeyError:
                pass

            try:
                concatenated_coco.dataset["info"] = dataset["info"]
            except KeyError:
                pass

            try:
                concatenated_coco.dataset["type"] = dataset["type"]
            except KeyError:
                pass

        # Specify the output directory for the concatenated dataset
        concatenated_output_dir = os.path.join(args.output_dir, "concatenated")
        os.makedirs(concatenated_output_dir, exist_ok=True)

        # Save the concatenated COCO dataset
        concatenated_json_file = os.path.join(
            concatenated_output_dir, "concatenated_coco.json"
        )
        with open(concatenated_json_file, "w") as f:
            json.dump(concatenated_coco.dataset, f, indent=2)

        print(f"\nConcatenated COCO dataset saved to: {concatenated_json_file}")


if __name__ == "__main__":
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
        "--tile-size", type=int, default=100, help="Tile width/height in meters."
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
        help="Pattern to match the vector file name. Defaults to _building.",
    )
    parser.add_argument(
        "--concatenate",
        action="store_true",
        help="Concatenate individual COCO datasets into one.",
    )
    parser.add_argument(
        "--info",
        help="Path to the info JSON file. Either leave empty, or put a folder path., containing info.json for each raster, with the same names as the rasters.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a batch job from an output directory.",
    )
    parser.add_argument(
        "--no-workers",
        type=int,
        default=1,
        help="Number of workers to use for parallel processing.",
    )

    args = parser.parse_args()
    main(args)
