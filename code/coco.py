import os
import json
from PIL import Image
from pathlib import Path
from collections import deque


class COCO:

    def __init__(self, images_path, captions_path):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        Mostly for preprocessing and preparing the data to be used later
        :param images_path (path): path of images folder
        :param captions_path (path): path of caption file
        """
        self.images_path = images_path
        self.captions_path = captions_path

        # load captions json file and put json['captions'] and json['images'] into attributes:
        self.captions, self.images = self._load_annotations()

        # create dictionary for images
        # where the image file name is the key and the corresponding captions are the item
        self.imgs_caps_dict = self._create_caption_dict()
    
    # method to load the json annotations file and extract the useful information from it
    def _load_annotations(self) -> list:
        # Opening JSON file
        with open(self.captions_path) as captions_json:
            # returns JSON object as
            # a dictionary
            data = json.load(captions_json)

        # annotations / Captions
        annotations = data['annotations']
        images_files = []
        images_info = data['images']
        for img in images_info:
            images_files.append(img['file_name'])
        return annotations, images_files

    # method to create dictionary of captions for each image
    # returns dict with img_file_name as key and captions as data
    def _create_caption_dict(self) -> dict:
        # create thd dict
        imgs_dict = dict()

        # loop through the imgs and store the names of the files as keys
        # the will be POSIX file path objects
        for img in self.images:
            imgs_dict[img] = []

        # loop through the caps dict and store each cap to the corresoanding file name or key
        for cap in self.captions:
            # the naming convention used in MS COCO is that each file name consists of 12 characters
            # zfill method pad the name with 0s to the left until we get 12 chars
            # img_file_name = self.imgs[0].parent / (str(cap['image_id']).zfill(12) + '.jpg')
            img_file_name = str(cap['image_id']).zfill(12) + '.jpg'
            # there are some imgs with more than 5 captions
            # just make sure all images are on the same ground with the same number of captions
            if len(imgs_dict[img_file_name]) < 5:
                imgs_dict[img_file_name].append(cap['caption'])

        return imgs_dict

    # method to return captions for a specific file
    def get_captions(self, file_name) -> list:
        assert file_name in self.imgs_caps_dict.keys(), "Can't find captions for this image"
        captions = self.imgs_caps_dict[file_name]

        return captions
    
    # method to return Image object, takes as input the index of the image
    def get_img(self, idx) -> Image:
        return Image.open(self.imgs_path / self.images[idx])

    '''
       method to convert imgs_caps_dict to list
       the purpose of this method is to create a list to use it when we want to create a vocabulary
       we will use deque object instead of list object, the reason why is because deque has O(1) for adding and removing objects from it
       while list will take a very long time add all the captions to it 
       in small datasets we can use list it won't be any problem, however with this dataset, my computer crached while trying to do that :)
       Stay safe and protect your machines
    '''
    def captions_to_list(self) -> deque:
        captions_deque = deque()
        for img_file_name in self.imgs_caps_dict:
            for cap in self.imgs_caps_dict[img_file_name]:
                captions_deque.append(cap)

        return captions_deque
