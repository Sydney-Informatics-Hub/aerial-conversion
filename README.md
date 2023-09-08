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

## Usage

```
python geojson2coco.py \
                --raster-file /path/to/data/chatswood_hd.tif \
                --polygon-file /path/to/data/chatswood.geojson \
                --tile-dir /path/to/data/big_tiles \
                --json-name /path/to/data/coco_from_gis_hd.json \
                --info /path/to/data/info.json \
                --class-column zone_name 

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