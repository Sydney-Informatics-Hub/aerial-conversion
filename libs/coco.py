import json
import pandas as pd
import geopandas as gpd


class coco_json: 
    """Class to hold the coco json format
    
    Attributes:
        coco_image (coco_image): coco_image object
        coco_images (coco_images): coco_images object
        coco_poly_ann (coco_poly_ann): coco_poly_ann object
        coco_poly_anns (coco_poly_anns): coco_poly_anns object
        
    """
    def toJSON(self):
        return(json.dumps(self, default=lambda o: o.__dict__, indent = 4))
    
    class coco_image: 
        pass
    
    class coco_images: 
        pass
        
    class coco_poly_ann: 
        pass
    
    class coco_poly_anns: 
        pass
    

def make_category(class_name:str, class_id:int, supercategory:str="landuse", trim = 0):
    """
    Function to build an individual COCO category

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
        "name": class_name[trim:]
    }
    return(category)



def make_category_object(geojson:gpd.GeoDataFrame, class_column:str, trim:int):
    """
    Function to build a COCO categories object.

    Args:
        geojson (gpd.GeoDataFrame): GeoDataFrame containing class data
        class_column (str): Name of column containing class names
        trim (int): Number of characters to trim from class name

    Returns:
        categories_json (list): List of COCO category objects
    """
    
    # TODO: Implement way to read supercategory data.

    supercategory = "landuse"
    classes = pd.DataFrame(geojson[class_column].unique(), columns = ["class"])
    classes['class_id'] = classes.index
    categories_json = []
    
    for _, row in classes.iterrows():
        categories_json.append(make_category(row["class"], row["class_id"], supercategory, trim))
    
    return(categories_json) 