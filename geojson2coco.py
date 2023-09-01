import argparse
import geopandas as gpd
import glob
import os.path
import rasterio as rio
import traceback
from pathlib import Path
import os

from libs.coordinates import wkt_parser, pixel_polygons_for_raster_tiles
from libs.tiles import save_tiles
from libs.coco import coco_json, coco_image_annotations, coco_polygon_annotations, make_category_object


def assemble_coco_json(raster_file_list, geojson, license_json, info_json, categories_json):
    
    pixel_poly_df = pixel_polygons_for_raster_tiles(raster_file_list, geojson)

    coco = coco_json()
    coco.images = coco_image_annotations(raster_file_list).images
    coco.annotations = coco_polygon_annotations(pixel_poly_df)
    coco.license = license_json
    coco.categories = categories_json
    coco.info = info_json
    
    return(coco)



#%% Command-line driver

def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polygon-file", required=True, default=".", type=Path)
    ap.add_argument("--raster-file", required=True, type=Path)
    ap.add_argument("--tile-size", default = 1000, type=int, help = "Int length in meters of square tiles to generate from raster. Defaults to 1000 meters.")
    ap.add_argument("--tile-dir", required = True, type = Path)
    ap.add_argument("--class-column", type = str, help = "Column name in GeoJSON where classes are stored.")
    ap.add_argument("--json-name", default="coco_from_gis.json", type=Path)
    ap.add_argument("--crs", type = str, default=None , help = "Specifiy the project crs to use.")
    ap.add_argument("--trim-class", default = 0, type = int, help = "Characters to trim of the start of each class name. A clummsy solution, set to 0 by default which leaves class names as is.")
    ap.add_argument("--cleanup", default = False, type = bool, help = "If set to true, will purge *.tif tiles from the directory. Default to false.")
    ap.add_argument("--save-gdf", default = True, type = bool, help = "If set to true, will save a GeoDataFrame that you can use to reconstruct a spatial version of the dataset.")
    ap.add_argument("--short-file-name", type = bool, help = "If True, saves a short file name in the COCO for images.")
    ap.add_argument("--license", type = Path, help = "Path to a license description in COCO JSON format. If not supplied, will default to MIT license.")
    ap.add_argument("--info", required = True, type = Path, help = "Path to info description in COCO JSON format.")
    args = ap.parse_args(args)

    """
    Create tiles from raster and convert to COCO JSON format.
    """

    print(f"Creating {args.tile_size} m*m tiles from {args.raster_file}")
    raster_path = args.raster_file
    geojson_path = args.polygon_file
    out_path = args.tile_dir
    tile_size = args.tile_size
    user_crs = args.crs

    # Read input files
    geotiff = rio.open(raster_path)
    geojson = gpd.read_file(geojson_path)

    # Reproject geojson on geotiff
    if user_crs is None:
        user_crs = geotiff.crs.to_wkt()
        user_crs = wkt_parser(user_crs)

    try:
        geojson = geojson.to_crs(user_crs)
    except Exception as e:
        print("CRS not recognized, please specify a valid CRS")
        traceback.print_exc()
        raise e
    
    # Create raster tiles
    save_tiles(geotiff, out_path, tile_size, tile_template="tile_{}-{}.tif")
    geotiff.close()

    # Read the created raster tiles into a list.
    raster_file_list = []
    for filename in glob.iglob(os.path.join(f"{out_path}","*.tif")):
        raster_file_list.append(filename)

    print(f"{len(raster_file_list)} raster tiles created")

    # Create class_id for category mapping
    geojson['class_id'] = geojson[args.class_column].factorize()[0]
    categories_json = make_category_object(geojson, args.class_column, args.trim_class)

    # If license is not supplied, use MIT by default
    if args.license is None:
        license_json = {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
            }
    else:
        # Read user supplied license
        # TODO: incorporate different licenses depending on images: this feature is almost never used but would be nice to support.
        license_json = open(args.license, 'r')

    info_json = open(args.info, 'r')

    print("Converting to COCO")
    # We are now ready to make the COCO JSON.
    spatial_coco = assemble_coco_json(raster_file_list, geojson, license_json, info_json, categories_json)

    # Write COCO JSON to file.
    with open (args.json_name, "w") as f:
        f.write(spatial_coco.toJSON())
    print(f"COCO JSON saved to {args.json_name}")
    
    # if args.save_gdf == True:
        
    #     pixel_poly_df['raster_tile_name'] = pixel_poly_df.apply(lambda row: os.path.basename(row['raster_tile']), axis = 1)
        
    #     with open ("gdf_output.csv", "w") as f:
    #         f.write(pixel_poly_df)
    
if __name__ == '__main__':
    main()
