import os
import json
from PIL import Image

class COCO:
    
    # constructor
    def __init__(self, root, imgs_path, captions_path):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param root (str): location of the data on the disk
        :param imgs_path (str): location of img folder
        :param captions_path (str): location to the captions folder.
        :return:
        """
        self.root = root
        self.imgs_path = imgs_path
        self.captions_path = captions_path
        
        self.imgs = os.listdir(os.path.join(self.root, self.imgs_path))
        self.captions = self._load_captions()
        self.imgs_caps_dict = self._create_caption_dict()
        
    
    # Load captions
    def _load_captions(self) -> list:
        # Opening JSON file
        captions_json = open(os.path.join(self.root,self.captions_path))

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
        for img in self.imgs:
            imgs_dict[img] = []
        
        # loop through the caps dict and store each cap to the corresoanding file name or key
        for cap in self.captions:
            # the naming convition used in MS COCO is that each file name consists of 12 charatcters
            # zfill method pad the name with 0s to the left until we get 12 chars
            img_file_name = str(cap['image_id']).zfill(12) + '.jpg'
            
            # there are some imgs with more than 5 captions
            # just make sure all images are on the same ground with the same number of captions
            if len(imgs_dict[img_file_name]) < 5:
                imgs_dict[img_file_name].append(cap['caption'])
        
        return imgs_dict
    
    
    # method to return an image as Image
    def get_img(self, idx):
        return Image.open(os.path.join(self.root, self.imgs_path, self.imgs[idx]))
    
    # method to return captions for a specific 
    def get_captions(self, file_name):
        assert file_name in self.imgs_caps_dict.keys(), "Can't find captions for this image"
        captions = self.imgs_caps_dict[file_name]
        
        return captions
        