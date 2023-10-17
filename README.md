# aerial-conversion
Open source tools enabling interchange between computer vision annotation and GIS data formats. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation

---

## Input and Output Data Formats

The repository can convert between the following formats:

* Images and annotations in COCO JSON format. 

* Georeferenced shapefile polygon vector files, with a readme file linking to the original web map server or aerial imagery source to be rendered.

---

## Setup

```
conda create -n aerial-conversion-dev python==3.9

conda activate aerial-conversion-dev

pip install -r requirements.txt
```



## Dataset

A toy dataset has been uploaded to Roboflow. It is a small subset, containing Chatswood region, available [here](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200).

There are multiple versions of this dataset. Please ignore the first version. Version 2 and later versions are the ones that are being used. The main difference of version 2 and 3 is that [version 2](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/2) contains 90 degree augmentaions, while [version 3](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/3) does not.

For implementing this in your code, you can use the following code snippet:

```python
from roboflow import Roboflow
 
rf = Roboflow(api_key= 'your_roboflow_api_key' )
workspace_name = "sih-vpfnf" 
dataset_version = 3 
project_name = "gis-hd-200x200" 
dataset_download_name = "coco-segmentation" 

project = rf.workspace(workspace_name).project(project_name)
dataset = project.version(dataset_version).download(dataset_download_name)
```
<!-- 
# Register the dataset
from detectron2.data.datasets import register_coco_instances
dataset_name = "chatswood-dataset" #@param {type:"string"}
dataset_folder = "gis-hd-200x200" #@param {type:"string"}
register_coco_instances(f"{dataset_name}_train", {}, f"{dataset_folder}/train/_annotations.coco.json", f"/content/{dataset_folder}/train/")
register_coco_instances(f"{dataset_name}_val", {}, f"{dataset_folder}/valid/_annotations.coco.json", f"/content/{dataset_folder}/valid/")
register_coco_instances(f"{dataset_name}_test", {}, f"{dataset_folder}/test/_annotations.coco.json", f"/content/{dataset_folder}/test/")

# Use the dataset
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
cfg.DATASETS.TEST = (f"{dataset_name}_test",)
# then do the other configs

``` -->



## Usage

To create tiles from a raster file, use the following command:


```
python geojson2coco.py \
                --raster-file /path/to/data/chatswood_hd.tif \
                --polygon-file /path/to/data/chatswood.geojson \
                --tile-dir /path/to/data/big_tiles \
                --json-name /path/to/data/coco_from_gis_hd.json \
                --info /path/to/data/info.json \
                --class-column zone_name 
```

To merge multiple COCO JSON files, and yield a geojson file for the input raster, use the following command:

```
python coco2geojson.py \
                /path/to/data/raster_tiles/dir \
                /path/to/data/predictions-coco.json \
                --tile-extension .tif \
                --geojson-output /path/to/data/output.geojson \
                --meta-name <name_of_the_dataset>
                --minimum-rotated-rectangle 
```
<!-- ---

## Documentation

The documentation for the project is provided in the [documentation](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/docs/_build/html/index.html) file.
Please read the [documentation](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/docs/_build/html/index.html) for further informationon the project, modules, and dependencies. -->

---

## Contributing to the Project

Please make sure to install all the required libraries in the [requirements.txt](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/requirements.txt) file for development.


### Commit rules:

In this project, `pre-commit` is being used. Hence, please make sure you have it in your env by `pip install pre-commit`.

Make sure to run pre-commit on each run. You can run it before commit on all files to check your code by typing `pre-commit run --all-files --verbose`, without staging your pre-commit config.
Otherwise, you can run it via `pre-commit run` only, or just envoking it while committing (and failing, and committing again).

Alternatively, to add the hook, after installing pre-commit, run:

```
pre-commit install
```
<!-- 
### Documentation update:

* To update the documentation, navigate to the [docs](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/docs/) directory.
* Remove the old `rst` files from the the docs directory, except `index.rst`.
* Navigate to the upper directory: `cd ..`.
* Input `sphinx-apidoc -o docs .` to regenerate the `rst` files.
* Navigate back to the `docs` directory.
* Update the `index.rst` file to include the new `rst` files, if required. Usually not needed. (You don't have to include the submodules.)
* Then input `make html` for updating the html file. -->