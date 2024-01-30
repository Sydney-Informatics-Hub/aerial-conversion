# aigis

`aigis` is a comprehensive toolkit for aerial and satellite imagery acquisition, processing, annotation, and analysis using artificial intelligence. This repository contains three main components:

1. **aigis annotate:** Tools for annotating aerial imagery data.
2. **aigis convert:** Utilities for converting aerial imagery data to various formats, including COCO, GeoJSON, etc.
3. **aigis segment:** Scripts and notebooks for segmenting aerial imagery using deep learning models.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/aigis.git
cd aigis
```

To work with each of the sub modules individually, navigate to their directories and install their dependencies:

```bash
python -m aerial_conversion.scripts.geojson2coco \
                --raster-file /path/to/data/chatswood_hd.tif \
                --polygon-file /path/to/data/chatswood.geojson \
                --tile-dir /path/to/data/big_tiles \
                --json-name /path/to/data/coco_from_gis_hd.json \
                --info /path/to/data/info.json \
                --class-column zone_name 
```

To merge multiple COCO JSON files, and yield a geojson file for the input raster, use the following command:

```bash
python -m aerial_conversion.scripts.coco2geojson \
                /path/to/data/raster_tiles/dir \
                /path/to/data/predictions-coco.json \
                --tile-extension tif \
                --geojson-output /path/to/data/output.geojson \
                --meta-name <name_of_the_dataset>
                --minimum-rotated-rectangle 
```

To do a batch conversion, when the conversion should be carried on multiple input images, use the following command:

```bash
python -m aerial_conversion.scripts.batch_geojson2coco \
                --raster-dir /path/to/data/rasters/ \
                --vector-dir /path/to/data/geojsons/ \
                --output-dir /path/to/data/outputs/ \
                --tile-size <size_of_the_tiles>
                --class-column <class_column_in_geojsons_to_look_for>
                --overlap <overlap_in_percentage_between_tiles> \
                --pattern <a_pattern_to_ensure_raster_names_matches_the_geojsons>
                --concatenate <whether_to_concatenate_the_output_geojsons_into_one_big_json>
                --info /path/to/data/info.json \
                --resume <whether_to_resume_the_process_in_case_new_images_are_added>
```

## Usage

### aigis annotate
This component provides scripts for annotating aerial imagery data. Detailed usage instructions can be found in the aerial_annotation directory.

### aigis convert
Aerial Conversion includes tools for converting aerial imagery data to various formats. For detailed instructions, refer to the aerial_conversion directory.

### aigis segment
Aerial Segmentation contains scripts and notebooks for segmenting aerial imagery using deep learning models. Refer to the aerial_segmentation directory for more details.

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github/Sydney-Informatics-Hub/aigis/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github/Sydney-Informatics-Hub/aigis/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github/Sydney-Informatics-Hub/aigis/issues)**: Submit bugs found or log feature requests for Aigis.



## Other Scrpts

### Splitting Dataset

Splitting dataset to train, test, and validation sets can be achieved using the following script:

```bash
python -m aerial_conversion.scripts.coco_split -s 0.7 /path/to/concatenated_coco.json /path/to/save/output/train.json /path/to/save/output/test_valid.json

python -m aerial_conversion.scripts.coco_split -s 0.667 /path/to/test_valid.json /path/to/save/output/test.json /path/to/save/output/valid.json
```


### Balancing dataset

To tinker with the dataset and balance it, the following scrips can be used. 

To isolate the categories:
    
```bash
python -m aerial_conversion.scripts.coco_balance -i /path/to/input/coco.json -o /path/to/output/coco-catlimited.json -c '<category 1>,<category 2>,...' --int_cats
```

`--int_cats` argument is a store-true argument. If it is set, the categories will be interpreted as integers. Otherwise, they will be interpreted as strings.

`-c` argument is the categories to be isolated. They should be comma separated.


To balance the dataset by removing a subsample of the images which have only a single category (the biggest category):

```bash
python -m aerial_conversion.scripts.coco_balance -i /path/to/input/coco.json -o /path/to/output/coco-balanced.json --balance_cats
```

`--balance_cats` argument is a store-true argument. If it is set, the dataset will be balanced by removing a subsample of the images which have only a single category (the biggest category).

