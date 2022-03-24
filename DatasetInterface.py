# import libraries

import os
from pathlib import Path
import numpy as np
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torchvision import transforms

from MSCOCO import COCO

# Interface Class
from Vocabulary import Vocabulary


class MSCOCOInterface(data.Dataset):

    # constructor
    def __init__(self, imgs_path, captions_path, freq_threshold, sequence_length, idx_to_string=None, string_to_index=None):
        """
        Constructor of MS COCO Interface for get imgs and caps as tensors.
        :param imgs_path (Pathlib object): location of img folder
        :param captions_path (Pathlib object): location to the captions folder.
        :param freq_threshold (int): if a word is not repeated enough don't add it to the dictionary
        :param sequence_length (int): all sequences must be the same lenght, so this param is used to pad or cut from sentence
        """

        self.captions_path = captions_path
        self.imgs_path = imgs_path
        self.freq_threshold = freq_threshold
        self.sequence_length = sequence_length
        
        
        self.coco = COCO(self.imgs_path, self.captions_path)
        #self.vocabulary = Vocabulary(self.freq_threshold, self.sequence_length, self.idx_to_string, self.string_to_index)
        
        if idx_to_string is None or string_to_index is None:
            self.vocabulary = Vocabulary(self.freq_threshold, self.sequence_length)
            self.create_vocabulary()
        
        else:
            self.string_to_index = string_to_index
            self.idx_to_string = idx_to_string
            self.vocabulary = Vocabulary(self.freq_threshold, self.sequence_length, self.idx_to_string, self.string_to_index)


    # method to create vocabulary
    def create_vocabulary(self):
        d = self.coco.captions_to_list()
        self.vocabulary.build_vocabulary(self.coco.captions_to_list())
        self.idx_to_string = self.vocabulary.idx_to_string
        self.string_to_index = self.vocabulary.string_to_index

    # image transforms
    def img_transforms(self, img):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transformer(img)

    # load image as Image then transform it to tensor
    def load_img(self, idx):
        img = self.coco.get_img(idx)
        img = self.img_transforms(img)
        return img

    def get_captions(self, img_file_name):
        return self.coco.get_captions(img_file_name)

    def display_img_with_captions(self, idx):
        img_file_name = self.coco.imgs[idx]
        img = self.coco.get_img(idx)
        caps = self.get_captions(img_file_name)
        plt.imshow(img)
        plt.show()
        print(caps[0])

    # return the length of the dataset
    def __len__(self):
        return len(self.imgs)
    
    # get an item from the dataset
    def __getitem__(self, idx):       
        img_file_name = self.coco.imgs[idx]
        # get X: Image
        X = self.load_img(idx)
        # get y: Image Caption
        y = self.get_captions(img_file_name)[0]
        y = self.vocabulary.numericalize(y)
        y = torch.tensor(y, dtype=torch.int64)
        return idx, X, y
