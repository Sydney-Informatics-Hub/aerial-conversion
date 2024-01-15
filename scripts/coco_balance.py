# -*- coding: utf-8 -*-
# from pycocotools.coco import COCO
import argparse
import json

# from tqdm import tqdm
import pandas as pd

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


def main(args=None):
    if args is None:
        args = parse_arguments(args)

    # Load the COCO json file
    with open(args.json_path, "r") as f:
        json_data = json.load(f)

    stats(json_data)


def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Balance a COCO dataset")
    parser.add_argument(
        "--json_path", "-i", type=str, help="Path to your COCO json file"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, help="Path to your output COCO json file"
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
