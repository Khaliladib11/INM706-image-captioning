import os
import json
from PIL import Image

class COCO:
    
    # constructor
    def __init__(self, imgs, captions_path):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param imgs (path): sorted list of image file names
        :param captions_path (path): path of captions file
        :return:
        """
        self.imgs = imgs
        self.captions_path = captions_path
        
        self.captions = self._load_captions()
        self.imgs_caps_dict = self._create_caption_dict()
        
    
    # Load captions
    def _load_captions(self) -> list:
        # Opening JSON file
        captions_json = open(self.captions_path)

        # returns JSON object as
        # a dictionary
        data = json.load(captions_json)
        data = data['annotations']
        return data
    
    
    # method to create dictionary of captions for each image
    # returns dict with img_file_name as key and captions as data
    def _create_caption_dict(self) -> dict:
        # create thd dict
        imgs_dict = dict()
        
        # loop through the imgs and store the names of the files as keys
        # the will be POSIX file path objects
        for img in self.imgs:
            imgs_dict[img] = []
        
        # loop through the caps dict and store each cap to the corresoanding file name or key
        for cap in self.captions:
            # the naming convention used in MS COCO is that each file name consists of 12 characters
            # zfill method pad the name with 0s to the left until we get 12 chars
            img_file_name = self.imgs[0].parent / (str(cap['image_id']).zfill(12) + '.jpg')
            
            # there are some imgs with more than 5 captions
            # just make sure all images are on the same ground with the same number of captions
            if len(imgs_dict[img_file_name]) < 5:
                imgs_dict[img_file_name].append(cap['caption'])
        
        return imgs_dict
    
    
    # method to return an image as Image
    def get_img(self, idx):
        return Image.open(self.imgs[idx])
    
    # method to return captions for a specific file
    def get_captions(self, file_name):
        assert file_name in self.imgs_caps_dict.keys(), "Can't find captions for this image"
        captions = self.imgs_caps_dict[file_name]
        
        return captions
        