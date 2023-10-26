# -*- coding: utf-8 -*-
"""This script supports batch conversion of paired geojson and raster data into a series of COCO datasets."""

import os
import subprocess
import argparse
import shutil
from pycocotools.coco import COCO

def main(args):
    # Specify the output directory
    output_dir = args.output_dir

    # Iterate over the raster directory
    for raster_file in os.listdir(args.raster_dir):
        # Check if the file is a GeoTIFF
        if raster_file.endswith('.tif'):
            # Get the file name without extension
            file_name = os.path.splitext(raster_file)[0]

            # Construct the vector file name
            vector_file = file_name + '.geojson'
            vector_path = os.path.join(args.vector_dir, vector_file)

            # Check if the vector file exists
            if os.path.exists(vector_path):
                # Specify the output directory for the file pair
                pair_output_dir = os.path.join(output_dir, file_name)
                os.makedirs(pair_output_dir, exist_ok=True)

                # Specify the output JSON file path
                json_file = os.path.join(pair_output_dir, 'coco_from_gis.json')

                # Construct the command
                command = [
                    'python', 'scripts/geojson2coco.py',
                    '--raster-file', os.path.join(args.raster_dir, raster_file),
                    '--polygon-file', vector_path,
                    '--tile-dir', pair_output_dir,
                    '--json-name', json_file,
                    '--info', os.path.join(pair_output_dir, 'info.json'),
                    '--tile-size', str(args.tile_size),
                    '--class-column', args.class_column
                ]

                try:
                    # Run the command
                    subprocess.run(command, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {vector_file}: {e.stderr}")
    
    # Generate markdown output
    print('Running geojson2coco.py over raster and vector pairs:')
    print()
    print('| Raster File | Vector File | JSON File |')
    print('|-------------|-------------|-----------|')
    for pair_dir in os.listdir(output_dir):
        pair_output_dir = os.path.join(output_dir, pair_dir)
        if os.path.isdir(pair_output_dir):
            raster_file = pair_dir + '.tif'
            vector_file = pair_dir + '.geojson'
            json_file = os.path.join(pair_output_dir, 'coco_from_gis.json')
            print(f'| {raster_file} | {vector_file} | {json_file} |')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raster and vector pairs to COCO JSON format.")
    parser.add_argument("--raster-dir", required=True, help="Path to the raster directory.")
    parser.add_argument("--vector-dir", required=True, help="Path to the vector directory.")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory.")
    parser.add_argument("--tile-size", type=int, default=100, help="Tile size in meters.")
    parser.add_argument("--class-column", default="trees", help="Column name in GeoJSON for classes.")
    
    args = parser.parse_args()
    main(args)
