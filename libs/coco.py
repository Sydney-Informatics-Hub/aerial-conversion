# -*- coding: utf-8 -*-
import json
import logging
import os

import geopandas as gpd
import pandas as pd
import rasterio as rio
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


class coco_json:
    """Class to hold the coco json format.

    Attributes:
        coco_image (coco_image): coco_image object
        coco_images (coco_images): coco_images object
        coco_poly_ann (coco_poly_ann): coco_poly_ann object
        coco_poly_anns (coco_poly_anns): coco_poly_anns object
    """

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    class coco_image:
        pass

    class coco_images:
        pass

    class coco_poly_ann:
        pass

    class coco_poly_anns:
        pass


def make_category(
    class_name: str, class_id: int, supercategory: str = "landuse", trim=0
):
    """Function to build an individual COCO category.

    Args:
        class_name (str): Name of class
        class_id (int): ID of class
        supercategory (str, optional): Supercategory of class. Defaults to "landuse".
        trim (int, optional): Number of characters to trim from class name. Defaults to 0.

    Returns:
        category (dict): COCO category object
    """

    category = {
        "supercategory": supercategory,
        "id": int(class_id),
        "name": class_name[trim:],
    }
    return category


def make_category_object(geojson: gpd.GeoDataFrame, class_column: str, trim: int):
    """Function to build a COCO categories object.

    Args:
        geojson (gpd.GeoDataFrame): GeoDataFrame containing class data
        class_column (str): Name of column containing class names
        trim (int): Number of characters to trim from class name

    Returns:
        categories_json (list): List of COCO category objects
    """

    # TODO: Implement way to read supercategory data.

    supercategory = "landuse"
    classes = pd.DataFrame(geojson[class_column].unique(), columns=["class"])
    classes["class_id"] = classes.index
    categories_json = []

    for _, row in classes.iterrows():
        categories_json.append(
            make_category(row["class"], row["class_id"], supercategory, trim)
        )

    return categories_json


def raster_to_coco(
    raster_path: str,
    index: int,
    extension: str = "png",
    bands: int = 3,
    colour: bool = True,
):
    """Function to convert a raster to a COCO image object.

    Args:
        raster_path (str): Path to raster
        index (int): Index of raster
        extension (str, optional): Extension of image. Defaults to "png".
        bands (int): The number of bands to save. Default is 3 (for R-G-B).
        colour (bool, optional): If True, save image in colour. Defaults to True.

    Returns:
        image (coco_image): COCO image object
    """

    geotiff = rio.open(raster_path)
    raster = geotiff.read()
    # print(raster[:3].shape)
    if bands > 1:
        raster = raster[:bands]
        if colour is False:
            # take the average of raster bands
            raster = raster.mean(axis=0)
            bands = 1
    # print(raster.shape)
    raster_name = os.path.splitext(raster_path)[0]
    image_name = f"{raster_name}.{extension}"

    with rio.Env():
        with rio.open(
            image_name,
            "w",
            driver=extension.upper(),
            height=geotiff.shape[0],
            width=geotiff.shape[1],
            count=bands,
            dtype=geotiff.dtypes[0],
            nodata=0,
            compress="deflate",
        ) as dst:
            if colour is False:
                dst.write(raster, 1)
            else:
                dst.write(raster)

    # Create each individual image object
    image = coco_json.coco_image()
    image.license = 1
    image.file_name = os.path.basename(image_name)
    image.height = raster.shape[0]
    image.width = raster.shape[1]
    image.id = index

    return image


def coco_bbox(polygon):
    """Generate a COCO format bounding box from a Polygon.

    Based on code from:
    #https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format

    Args:
        polygon (Polygon): Polygon object

    Returns:
        cc_bbox (list): COCO format bounding box
    """

    bounds = polygon.bounds
    top_left_x = bounds[0]
    top_left_y = bounds[1]  # lowest y val, cause it's from top down.
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    cc_bbox = [top_left_x, top_left_y, width, height]

    return cc_bbox


def coco_polygon_annotation(pixel_polygon: list, image_id, annot_id, class_id):
    """Function to convert a polygon to a COCO annotation object.

    Args:
        pixel_polygon (list): List of pixel coordinates generated via coordinates.spatial_polygon_to_pixel_rio()
        image_id (int): ID of image
        annot_id (int): ID of annotation
        class_id (int): ID of class

    Returns:
        annot (dict): COCO annotation object
    """

    annot = {
        "segmentation": [item for sublist in pixel_polygon for item in sublist],
        "area": Polygon(pixel_polygon).area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": coco_bbox(Polygon(pixel_polygon)),
        "category_id": class_id,
        "id": annot_id,
    }

    return annot


def coco_polygon_annotations(polygon_df):
    """Function to convert a GeoDataFrame of polygons to a list of COCO
    annotation objects.

    Args:
        polygon_df (gpd.GeoDataFrame): GeoDataFrame containing polygon data

    Returns:
        annotations_tmp (list): List of COCO annotation objects
    """

    annotations_tmp = []
    for _, row in polygon_df.iterrows():
        annotations_tmp.append(
            coco_polygon_annotation(
                row["pixel_polygon"], row["image_id"], row["annot_id"], row["class_id"]
            )
        )

    return annotations_tmp


def coco_image_annotations(raster_file_list, colour):
    """Function to convert a list of rasters to a list of COCO image objects.

    Args:
        raster_file_list (list): List of raster files

    Returns:
        images (coco_images): coco_images object
    """

    images = coco_json.coco_images()
    images.images = [
        raster_to_coco(raster_file, ind, "png", 3, colour)
        for ind, raster_file in enumerate(raster_file_list)
    ]

    return images
