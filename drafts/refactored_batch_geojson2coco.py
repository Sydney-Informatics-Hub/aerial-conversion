import argparse
import json
import os
import pickle
import subprocess
import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm


def crop_and_save_geojson(raster_dir, geojson_path, raster_extension=".tif"):
    """Crop a GeoJSON file to the extent of a raster file and save it.

    Args:
        raster_dir (str): Path to the directory containing the raster files.
        geojson_path (str): Path to the GeoJSON file.
        raster_extension (str, optional): Extension of the raster files. Defaults to '.tif'.
    """

    # Read the GeoJSON file
    geojson = gpd.read_file(geojson_path)

    # Loop through each raster file
    for raster_file in os.listdir(raster_dir):
        if raster_file.endswith(raster_extension):
            raster_path = os.path.join(raster_dir, raster_file)

            # Open the raster file and get its bounds
            with rasterio.open(raster_path) as src:
                left, bottom, right, top = src.bounds

            # Create a bounding box from the raster bounds
            bbox = box(left, bottom, right, top)

            # Crop the GeoJSON to the extent of the raster
            cropped_geojson = geojson[geojson.geometry.intersects(bbox)]

            # Save the cropped GeoJSON with the same naming pattern
            cropped_geojson_filename = (
                os.path.dirname(geojson_path)
                + "/"
                + os.path.basename(raster_file).split(".")[0]
                + ".geojson"
            )
            cropped_geojson.to_file(cropped_geojson_filename, driver="GeoJSON")


def get_processed(output_dir):
    """
    Retrieve the list of directories already processed into COCO format.

    Args:
        output_dir (str): Path to the output directory.

    Returns:
        list: A list of directory names that are already processed.
    """
    processed = []
    for sub_dir in os.listdir(output_dir):
        json_path = os.path.join(output_dir, sub_dir, "coco_from_gis.json")
        if os.path.isfile(json_path):
            processed.append(sub_dir)
    return processed

def run_conversion(raster_dir, vector_dir, output_dir, resume=False, **kwargs):
    """
    Run the batch conversion of raster and geojson vector data into COCO datasets.

    Args:
        raster_dir (str): Path to the raster data directory.
        vector_dir (str): Path to the vector data (geojson) directory.
        output_dir (str): Path where COCO datasets will be saved.
        resume (bool): Whether to resume a previously incomplete job.

    Additional kwargs supported are:
        tile_size (int): The size of the tile in meters.
        class_column (str): The column to use as class names from geojson.
        overlap (float): The overlap between tiles as a percentage.
        pattern (str): The pattern to match geojson filenames with raster files.
        info (str): Path to the directory containing info JSON files.

    Returns:
        list: A list of paths to individual COCO dataset files.

    Raises:
        NotImplementedError: If parallel processing is enabled (not currently supported).
    """
    if kwargs.get('no_workers', 1) > 1:
        raise NotImplementedError("Parallel processing not implemented.")

    processed = get_processed(output_dir) if resume else []
    coco_datasets = []
    errors = {}

    # Process each raster file.
    for raster_file in os.listdir(raster_dir):
        if not raster_file.endswith(".tif"):
            continue
        
        file_name = os.path.splitext(raster_file)[0]
        if file_name in processed:
            print(f"Skipping {file_name} as it is already processed.")
            coco_datasets.append(os.path.join(output_dir, file_name, "coco_from_gis.json"))
            continue

        vector_file = f"{file_name}{kwargs.get('pattern', '')}.geojson"
        vector_path = os.path.join(vector_dir, vector_file)
        if not os.path.exists(vector_path):
            continue

        # Specify output paths and prepare directories.
        pair_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(pair_output_dir, exist_ok=True)
        
        json_file = os.path.join(pair_output_dir, "coco_from_gis.json")
        info_file = kwargs.get('info', os.path.join(output_dir, "info.json"))
        if not os.path.isfile(info_file):
            with open(info_file, "w") as f:
                json.dump({}, f)

        overlap = kwargs.get('overlap', 0)
        tile_size = kwargs.get('tile_size', 100)
        class_column = kwargs.get('class_column')
        if class_column is None:
            raise ValueError("class_column argument is required.")

        # Prepare and run the conversion command.
        command = [
            "geojson2coco",
            "--raster-file", os.path.join(raster_dir, raster_file),
            "--polygon-file", vector_path,
            "--tile-dir", pair_output_dir,
            "--json-name", json_file,
            "--offset", str(overlap),
            "--info", info_file,
            "--tile-size", str(tile_size),
            "--class-column", class_column,
        ]

        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {vector_file}: {e.stderr}")
            errors[file_name] = e.stderr
            error_df = pd.DataFrame.from_dict({file_name: e.stderr}, orient="index", columns=["error_message"])
            error_df.to_csv(os.path.join(output_dir, "error.csv"), mode='a')
        else:
            coco_datasets.append(json_file)

    # Save the errors and return the path to COCO datasets.
    with open(os.path.join(output_dir, "error.pkl"), "wb") as f:
        pickle.dump(errors, f)

    return coco_datasets

def concatenate_coco_datasets(coco_datasets, output_dir):
    """
    Concatenate individual COCO datasets into a single COCO dataset.

    Args:
        coco_datasets (list): A list of paths to individual COCO dataset files.
        output_dir (str): The directory to save the concatenated COCO dataset.

    Returns:
        str: The path to the concatenated COCO dataset.
    """
    concatenated = COCO(initialize_empty=True)
    category_index = image_index = annotation_index = 0
    for coco_file in tqdm(coco_datasets):
        with open(coco_file, 'r') as f:
            dataset = json.load(f)
        for image in dataset['images']:
            image['file_name'] = os.path.join(os.path.basename(os.path.dirname(coco_file)), image['file_name'])
            image['id'] = image_index
            concatenated.addImage(image)
            image_index += 1
        for ann in dataset['annotations']:
            ann['image_id'] = image_index - 1
            ann['id'] = annotation_index
            concatenated.addAnns([ann])
            annotation_index += 1
        for cat in dataset['categories']:
            if not concatenated.hasCat(cat['name']):
                cat['id'] = category_index
                concatenated.addCats([cat])
                category_index += 1
            else:
                cat['id'] = concatenated.getCatId(cat['name'])
            
    concatenated_output_dir = os.path.join(output_dir, "concatenated")
    os.makedirs(concatenated_output_dir, exist_ok=True)
    concatenated_json_file = os.path.join(concatenated_output_dir, "concatenated_coco.json")
    with open(concatenated_json_file, 'w') as f:
        json.dump(concatenated.dataset, f, indent=2)

    return concatenated_json_file

def main():
    parser = argparse.ArgumentParser(description="Convert raster and vector pairs to COCO JSON format.")
    parser.add_argument("--raster-dir", required=True, help="Path to the raster directory.")
    parser.add_argument("--vector-dir", required=True, help="Path to the vector directory.")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory.")
    parser.add_argument("--tile-size", type=int, default=100, help="Tile width/height in meters.")
    parser.add_argument("--class-column", required=True, help="Column name in GeoJSON for classes.")
    parser.add_argument("--overlap", default=0, help="Overlap between tiles in percentage.")
    parser.add_argument("--pattern", default="", help="Pattern to match the vector file names.")
    parser.add_argument("--concatenate", action="store_true", help="Concatenate individual COCO datasets into one.")
    parser.add_argument("--info", help="Path to the info JSON file.")
    parser.add_argument("--resume", action="store_true", help="Resume a batch job from an output directory.")
    parser.add_argument("--no-workers", type=int, default=1, help="Number of workers to use for parallel processing.")
    args = parser.parse_args()

    # Run conversion process
    coco_datasets = run_conversion(args.raster_dir, args.vector_dir, args.output_dir, resume=args.resume, **vars(args))
    print("Conversion completed.")

    # Optional concatenation of COCO datasets
    if args.concatenate:
        concatenated_json_file = concatenate_coco_datasets(coco_datasets, args.output_dir)
        print(f"Concatenated COCO dataset saved to: {concatenated_json_file}")

if __name__ == "__main__":
    main()