# -*- coding: utf-8 -*-
# from pycocotools.coco import COCO
import argparse
import json

import pandas as pd
from tqdm import tqdm

# import os


def stats(json_data):
    # categories = json_data["categories"]
    annotations = json_data["annotations"]
    # images = json_data["images"]
    # info = json_data["info"]
    # licenses = json_data["licenses"]

    annotation_list = []
    for annot in annotations:
        annotation_list.append(
            {"category_id": annot["category_id"], "image_id": annot["image_id"]}
        )
    annotation_df = pd.DataFrame(annotation_list)

    cats_per_image = annotation_df.groupby("image_id").count()
    cats_unique_per_image = annotation_df.groupby("image_id").nunique()
    cats_stat_per_image = annotation_df.groupby("category_id").count()

    print(cats_per_image)
    print(cats_unique_per_image)
    print(cats_stat_per_image)

    cats_unique_per_image = cats_unique_per_image.reset_index()
    cats_unique_per_image = cats_unique_per_image.groupby("category_id").count()
    cats_unique_per_image = cats_unique_per_image.reset_index()
    cats_unique_per_image.columns = ["cats", "images"]
    print("stats for the diversity of cats in images")
    print(cats_unique_per_image)


def isolate_cat(json_data: dict, cat_ids: list):
    # Limit the annotations to the given categories
    print("Limiting the annotations to the given categories")
    annotations = json_data["annotations"]
    new_annotations = []
    for annot in tqdm(annotations):
        if annot["category_id"] in cat_ids:
            new_annotations.append(annot)
    print("Removed {} annotations".format(len(annotations) - len(new_annotations)))
    assert len(new_annotations) > 0, "No annotations left after filtering"

    json_data["annotations"] = new_annotations

    # Limit the categories to the given categories
    print("Limiting the categories to the given categories")
    categories = json_data["categories"]
    new_categories = []
    for cat in tqdm(categories):
        if cat["id"] in cat_ids:
            new_categories.append(cat)
    print("Removed {} categories".format(len(categories) - len(new_categories)))
    json_data["categories"] = new_categories

    # Remove the images if not referenced in annotations
    print("Removing the images if not referenced in annotations")
    image_ids = []
    for annot in annotations:
        image_ids.append(annot["image_id"])
    image_ids = list(set(image_ids))
    images = json_data["images"]
    new_images = []
    for image in tqdm(images):
        if image["id"] in image_ids:
            new_images.append(image)
    print("Removed {} images".format(len(images) - len(new_images)))
    json_data["images"] = new_images

    return json_data


def main(args=None):
    if args is None:
        args = parse_arguments(args)

    # Load the COCO json file
    print("Loading the COCO json file")
    with open(args.json_path, "r") as f:
        json_data = json.load(f)

    # stats(json_data)
    if args.isolate_cat:
        cats = args.isolate_cat.split(",")
        if args.int_cats:
            cats = [int(cat) for cat in cats]
        json_data = isolate_cat(json_data, cats)

    # Save the COCO json file
    print("Saving the COCO json file")
    with open(args.output_path, "w") as f:
        json.dump(json_data, f, indent=2)


def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Balance a COCO dataset")
    parser.add_argument(
        "--json_path", "-i", type=str, help="Path to your COCO json file"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, help="Path to your output COCO json file"
    )
    parser.add_argument(
        "--isolate_cat",
        "-c",
        type=str,
        help="Comma separated list of category ids to isolate from to main dataset.",
    )
    parser.add_argument(
        "--int_cats",
        action=argparse.BooleanOptionalAction,
        help="Set this flag if the categores are integers.",
    )
    parser.add_argument(
        "--augmentation_low",
        "-a",
        type=float,
        default=0,
        help="Augmentation factor on low balance classes",
    )
    parser.add_argument(
        "--augmentation_high",
        "-b",
        type=float,
        default=0,
        help="Augmentation factor on high balance classes",
    )
    parser.add_argument(
        "--balance_remove_factor",
        "-r",
        type=float,
        default=0,
        help="Remove factor on high balance classes",
    )
    parser.add_argument(
        "--balance_full",
        action=argparse.BooleanOptionalAction,
        help="Smart balance the dataset. Will try to balance the dataset as much as possible by reducing the frequent classes after augmenting the less frequent classes.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        help="Level of noise to add to the dataset",
    )
    parser.add_argument(
        "--flip",
        action=argparse.BooleanOptionalAction,
        help="Flip the images for augmenttion",
    )
    parser.add_argument(
        "--rotate_min",
        type=float,
        default=0,
        help="Rotate the images for augmenttion by at least this amound in degrees.",
    )
    parser.add_argument(
        "--rotate_max",
        type=float,
        default=0,
        help="Rotate the images for augmenttion by up to this amound in degrees. If zero, will not rotate the images.",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
