# -*- coding: utf-8 -*-
"""This script supports batch conversion of paired geojson and raster data into
a series of COCO datasets."""
import argparse
import json
import os
import subprocess

from pycocotools.coco import COCO


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
    output_dir = args.output_dir

    individual_coco_datasets = []  # List to store individual COCO datasets

    # Iterate over the raster directory
    for raster_file in os.listdir(args.raster_dir):
        # Check if the file is a GeoTIFF
        if raster_file.endswith(".tif"):
            # Get the file name without extension
            file_name = os.path.splitext(raster_file)[0]

            # Construct the vector file name
            vector_file = file_name + ".geojson"
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

                # Add the generated COCO dataset to the list
                individual_coco_datasets.append(json_file)

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
        for coco_file in individual_coco_datasets:
            with open(coco_file, "r") as f:
                dataset = json.load(f)
                concatenated_coco.dataset.update(dataset)

        # Specify the output directory for the concatenated dataset
        concatenated_output_dir = os.path.join(args.output_dir, "concatenated")
        os.makedirs(concatenated_output_dir, exist_ok=True)

        # Save the concatenated COCO dataset
        concatenated_json_file = os.path.join(
            concatenated_output_dir, "concatenated_coco.json"
        )
        with open(concatenated_json_file, "w") as f:
            json.dump(concatenated_coco.dataset, f)

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
        "--class-column", default="trees", help="Column name in GeoJSON for classes."
    )
    parser.add_argument(
        "--overlap",
        default=0,
        help="Overlap between tiles in percentage. Defaults to 0.",
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

    args = parser.parse_args()
    main(args)
