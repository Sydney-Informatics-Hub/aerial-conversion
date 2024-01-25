# -*- coding: utf-8 -*-
import argparse
import glob
import os
import re
import warnings

import geopandas
import rasterio

# Get rid of the annoying warning every time a jpeg is opened in rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(
            description="Convert a directory of JPEG aerial images into GeoTIFF format"
            "given a GeoJSON containing image boundaries."
        )
        parser.add_argument(
            "input_dir",
            type=str,
            help="Directory containig jpeg images. The ID of the image to find in the"
            "GeoJSON should be the initial numeric characters in the input filename.",
        )
        parser.add_argument(
            "input_geojson",
            type=str,
            help="Input GeoJSON file containing tile boundaries. The row in the GeoJSON with ID derived"
            "from the initial numeric characters of the input filename will be searched for the tile boundaries.",
        )
        parser.add_argument(
            "--output_tiff_dir",
            type=str,
            default="./tiff_tiles",
            help="Name of output directory for converted TIFF format images.",
        )
        return parser.parse_args()

    args = parse_arguments()

    # Make the output directory if necessary
    os.makedirs(args.output_tiff_dir, exist_ok=True)

    # Get all the JPEGs in the input dir
    all_files = glob.glob(os.path.join(args.input_dir, "*"))
    jpeg_files = [
        file
        for file in all_files
        if os.path.splitext(file)[1].lower() in (".jpg", "jpeg")
    ]
    # Get GeoJSON as a GeoPandas DataFrame
    gdf = geopandas.read_file(args.input_geojson)

    for jpeg_file in jpeg_files:
        filename = os.path.splitext(os.path.basename(jpeg_file))[0]
        output_file = os.path.join(args.output_tiff_dir, filename + ".tif")

        # Get leading digits in the string and bail out if we have nothing
        numeric_id_list = re.findall(r"^\d+", filename)
        if len(numeric_id_list) < 1:
            print(f"No ID found for {jpeg_file}. Skipping ...")
            continue
        id = int(numeric_id_list[0])

        # Check we have a row in the GeoJSON for our ID and bail out if not
        desired_row = gdf[gdf.id == id]
        if len(desired_row) == 0:
            print(
                f"Cannot find id: {id} in {os.path.basename(args.input_geojson)}. Skipping ..."
            )
            continue
        # BBox for rasterio.transform should be (west, south, east, north)
        bbox = [
            desired_row.iloc[0][edge] for edge in ["left", "bottom", "right", "top"]
        ]

        dataset = rasterio.open(jpeg_file, "r")
        # Assuming RGB here
        bands = [1, 2, 3]
        data = dataset.read(bands)
        transform = rasterio.transform.from_bounds(*bbox, data.shape[2], data.shape[1])
        crs = {"init": str(gdf.crs)}
        # Write the GeoTiff
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            width=data.shape[2],
            height=data.shape[1],
            count=3,
            dtype=data.dtype,
            nodata=0,
            transform=transform,
            crs=crs,
        ) as dst:
            print(f"{jpeg_file} -> {output_file}")
            dst.write(data, indexes=bands)


if __name__ == "__main__":
    main()
