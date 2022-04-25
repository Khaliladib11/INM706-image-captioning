# import libraries
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torchvision import transforms

from MSCOCO import COCO
from Vocabulary import Vocabulary
from collections import deque
from PIL import Image
from pathlib import Path


# Interface Class
class MSCOCOInterface(data.Dataset):

    # constructor
    def __init__(self, imgs_path, captions_path, freq_threshold, sequence_length, caps_per_img=1, idx_to_string=None,
                 string_to_index=None):
        """
        Constructor of MS COCO Interface for get imgs and caps as tensors.
        :param imgs_path (Pathlib object): location of img folder
        :param captions_path (Pathlib object): location to the caption folder.
        :param freq_threshold (int): if a word is not repeated enough don't add it to the dictionary
        :param sequence_length (int): all sequences must be the same length, so this param is used to pad or cut from sentence
        :param caps_per_img (int): the number of image we want to return for each image
        :param idx_to_string (dict): dictionary contains the index to string vocabulary
        :param string_to_index (dict): dictionary contains the string to index vocabulary
        """

        self.captions_path = captions_path
        self.imgs_path = imgs_path
        self.freq_threshold = freq_threshold
        self.sequence_length = sequence_length
        if caps_per_img > 5:
            self.caps_per_img = 5
        elif caps_per_img < 1:
            self.caps_per_img = 1
        else:
            self.caps_per_img = caps_per_img

        self.coco = COCO(self.imgs_path, self.captions_path)
        # self.vocabulary = Vocabulary(self.freq_threshold, self.sequence_length, self.idx_to_string, self.string_to_index)

        if idx_to_string is None or string_to_index is None:
            self.vocabulary = Vocabulary(self.freq_threshold, self.sequence_length)
            self.create_vocabulary()

        else:
            self.string_to_index = string_to_index
            self.idx_to_string = idx_to_string
            self.vocabulary = Vocabulary(self.freq_threshold, self.sequence_length, self.idx_to_string,
                                         self.string_to_index)

        self.__create_data_deque()

    # method to create the list deque of imgs and captions accordind to the number of caption per image
    def __create_data_deque(self):
        self.img_deque = deque()
        imgs = self.coco.imgs
        for img in imgs:
            counter = 0
            while counter < self.caps_per_img:
                self.img_deque.append([img, self.get_captions(img)[counter]])
                counter += 1

    # method to create vocabulary
    def create_vocabulary(self):
        self.vocabulary.build_vocabulary(self.coco.captions_to_list())
        self.idx_to_string = self.vocabulary.idx_to_string
        self.string_to_index = self.vocabulary.string_to_index

    # image transforms
    def img_transforms(self, img):
        transformer = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transformer(img)

    # load image as Image then transform it to tensor
    def load_img(self, idx):
        img_file_name = self.img_deque[idx][0]
        img = Image.open(self.imgs_path / img_file_name).convert('RGB')
        img = self.img_transforms(img)
        return img

    def get_captions(self, img_file_name):
        return self.coco.get_captions(img_file_name)

    def display_img_with_captions(self, idx):
        # img_file_name = self.img_deque[idx][0]
        img = Image.open(self.imgs_path / self.img_deque[idx][0])
        cap = self.img_deque[idx][1]
        plt.imshow(img)
        plt.show()
        #print(cap)

    # return the length of the dataset
    def __len__(self):
        return len(self.img_deque)

    # get an item from the dataset
    def __getitem__(self, idx):
        # get X: Image
        X = self.load_img(idx)
        # get y: Image Caption
        y = self.img_deque[idx][1]
        y = self.vocabulary.numericalize(y)
        y = torch.tensor(y, dtype=torch.int64)
        return idx, X, y
