# import libraries
import matplotlib.pyplot as plt
import itertools
import json
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import nltk

from MSCOCO import COCO
from Vocabulary import Vocabulary
from collections import deque
from PIL import Image
from pathlib import Path

def get_loader(imgs_path,
               captions_path,
               freq_threshold,
               # sequence_length,  # might not need this
               batch_size=1,
               caps_per_img=1,
               vocab_from_file=True):
    """Returns the data loader
    :param imgs_path (Pathlib object): location of img folder
    :param captions_path (Pathlib object): location to the caption folder.
    :param freq_threshold (int): if a word is not repeated enough don't add it to the dictionary
    :param sequence_length (int): all sequences must be the same length, so this param is used to pad or cut from sentence
    :param caps_per_img (int): the number of image we want to return for each image
    :param vocab_from_file (bool): if true then load vocab from json files in vocabulary folder - currently no ability to specify file name
    """
    
    interface = MSCOCOInterface(imgs_path=imgs_path,
                                captions_path=captions_path,
                                freq_threshold=freq_threshold,
                                # sequence_length=sequence_length,  # might not need this
                                caps_per_img=caps_per_img,
                                vocab_from_file=vocab_from_file)
    
    # following lines create a batch of captions all of same length
    indices = interface.get_train_indices(batch_size)
    initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                            batch_size=batch_size,
                                            drop_last=False)
    data_loader = data.DataLoader(dataset=interface,
                                  batch_sampler=batch_sampler)
    return data_loader


# Interface Class
class MSCOCOInterface(data.Dataset):

    # constructor
    def __init__(self,
                 imgs_path,
                 captions_path,
                 freq_threshold,
                 # sequence_length,  # might not need this
                 caps_per_img=1,
                 vocab_from_file=True,  # new variable to load from json file
                 # idx_to_string=None,
                 # string_to_index=None,
                 ):
        """
        Constructor of MS COCO Interface for get imgs and caps as tensors.
        :param imgs_path (Pathlib object): location of img folder
        :param captions_path (Pathlib object): location to the caption folder.
        :param freq_threshold (int): if a word is not repeated enough don't add it to the dictionary
        :param sequence_length (int): all sequences must be the same length, so this param is used to pad or cut from sentence
        :param caps_per_img (int): the number of image we want to return for each image
        :param vocab_from_file (bool): if true then load vocab from json files in vocabulary folder - currently no ability to specify file name
        :param idx_to_string (dict): dictionary contains the index to string vocabulary
        :param string_to_index (dict): dictionary contains the string to index vocabulary
        """

        self.captions_path = captions_path
        self.imgs_path = imgs_path
        self.freq_threshold = freq_threshold
        # self.sequence_length = sequence_length
        if caps_per_img > 5:
            self.caps_per_img = 5
        elif caps_per_img < 1:
            self.caps_per_img = 1
        else:
            self.caps_per_img = caps_per_img

        self.rng = np.random.default_rng(42)
        self.coco = COCO(self.imgs_path, self.captions_path)

        # create vocab from scratch if it is not already done
        if vocab_from_file:
            p = Path('vocabulary')
            self.string_to_index = json.load(open(p/'string_to_index.json'))
            self.idx_to_string = json.load(open(p/'idx_to_string.json'))
            self.vocabulary = Vocabulary(self.freq_threshold,
                                         # self.sequence_length,
                                         self.idx_to_string,
                                         self.string_to_index)
            
        else:
            self.vocabulary = Vocabulary(self.freq_threshold, 
                                         # self.sequence_length
                                        )
            self.create_vocabulary()
            
        print("{}\nVocab size is {}\n{}".format("#"*20,
                                                len(self.idx_to_string),
                                                "#"*20))

        self.img_deque = self.__create_data_deque()
        
        # We need caption length with same index as caption in img_deque
        # to enable dataloader to load captions with the same length
        print('\nObtaining caption lengths...')
        all_tokens = [self.vocabulary.tokenizer_eng(str(self.img_deque[idx][1])) 
                      for idx in tqdm(range(len(self.img_deque)))]
        self.caption_lengths = [len(token) for token in all_tokens]

    # method to create the list deque of imgs and captions according to the number of caption per image
    # this will only create a list of captions for captions in the captions file - it doesn't matter
    # if more images have been loaded for the COCO dataset
    def __create_data_deque(self):
        img_deque = deque()
        imgs = self.coco.imgs
        for img in imgs:
            counter = 0
            while counter < self.caps_per_img:
                if len(self.get_captions(img)) > 0:
                    img_deque.append([img, self.get_captions(img)[counter]])
                    counter += 1
                else:
                    break
        return img_deque
    
    def get_train_indices(self, batch_size=1):
        sel_length = self.rng.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length 
                                for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=batch_size))
        return indices

    # method to create vocabulary
    def create_vocabulary(self):
        # coco.captions_to_list creates a list of (non-tokenized) captions
        self.vocabulary.build_vocabulary(self.coco.captions_to_list())
        self.idx_to_string = self.vocabulary.idx_to_string
        self.string_to_index = self.vocabulary.string_to_index

    # image transforms
    def img_transforms(self, img):
        transformer = transforms.Compose([
            transforms.Resize((256, 256)),
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
        # convert the image to RGB to make sure all the images are 3D, because there are some images in grayscale
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
        # print(cap)

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
