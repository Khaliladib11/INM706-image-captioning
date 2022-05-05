# import libraries
import matplotlib.pyplot as plt
import random
import itertools
import os

import torch
from torch.utils import data
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from coco import COCO
from vocabulary import Vocabulary
from collections import deque
from PIL import Image
from pathlib import Path
import random


class MSCOCODataset(data.Dataset):

    def __init__(self,
                 images_path,
                 captions_path,
                 freq_threshold,
                 caps_per_image,
                 mode='train',
                 transform=None,
                 idx2word=None, # here for backwards compatibility. We don't need it
                 word2idx=None
                 ):
        """
        Constructor for the dataset interface, inherited from torch.utils.data.Dataset

        :param images_path: the path to the images folder
        :param captions_path: path of caption file
        :param freq_threshold: if a word is not repeated enough don't add it to the dictionary
        :param caps_per_image: number of captions per image
        :param mode: to specify in which mode
        :param transform: torchvision.transforms object if there is a custom transform for images
        :param idx2word: loaded idx2word vocab
        :param word2idx: loaded word2idx vocab
        """

        self.images_path = images_path
        self.captions_path = captions_path
        self.freq_threshold = freq_threshold
        self.caps_per_image = caps_per_image
        self.transform = transform
        
        # initialize the COCO object
        self.coco = COCO(self.images_path, self.captions_path)
        
        # use fixed seed to select the same images each time
        random.seed(706)

        assert mode in ['train', 'validation', 'test'], "mode must be from ['train', 'validation', 'test']"
        assert word2idx is not None, "You have to use loaded word2idx vocab to train the model"

        self.word2idx = word2idx
        self.idx2word = dict(zip(word2idx.values(), word2idx.keys()))
        
#         # Training mode
#         if mode == 'train':
#             # take random 10_000 images from the first 20_000 images
#             imgs = deque(random.sample(self.coco.images[:20_000], 10_000))
#             caps = []
#             for img in imgs:
#                 for cp in self.coco.imgs_caps_dict[img]:
#                     caps.append(cp)
#             if idx2word is None and word2idx is None:
#                 self.vocab = Vocabulary(freq_threshold=self.freq_threshold)
#                 self.vocab.build_vocabulary(caps)
#             else:
#                 self.vocab = Vocabulary(freq_threshold=self.freq_threshold, idx2word=idx2word, word2idx=word2idx)

#         # validation mode
#         elif mode == 'validation':
#             # take random 2_000 images from the images between 20_001 and 25_000
#             imgs = deque(random.sample(self.coco.images[20_001:25_000], 2_000))
#             caps = []
#             for img in imgs:
#                 for cp in self.coco.imgs_caps_dict[img]:
#                     caps.append(cp)
#             assert idx2word is not None, "You have to use loaded idx2word vocab to validated the model"
#             assert word2idx is not None, "You have to use loaded word2idx vocab to validated the model"
#             self.vocab = Vocabulary(freq_threshold=self.freq_threshold, idx2word=idx2word, word2idx=word2idx)

#         # test mode
#         elif mode == 'test':
#             # take random 2_000 images from the images between 25_001 and 30_000
#             imgs = deque(random.sample(self.coco.images[25_001:30_001], 2_000))
#             caps = []
#             for img in imgs:
#                 for cp in self.coco.imgs_caps_dict[img]:
#                     caps.append(cp)
#             assert idx2word is not None, "You have to use loaded idx2word vocab to test the model"
#             assert word2idx is not None, "You have to use loaded word2idx vocab to test the model"
#             self.vocab = Vocabulary(freq_threshold=self.freq_threshold, idx2word=idx2word, word2idx=word2idx)

        # self.images = self.coco.images
        self.images = self.coco.images
        self.vocab = Vocabulary(freq_threshold=self.freq_threshold,
                                idx2word=self.idx2word,
                                word2idx=self.word2idx)
        self.__create_data_deque()

    # method to create the list deque of imgs and captions according to the number of caption per image
    # this will only create a list of captions for captions in the captions file - it doesn't matter
    # if more images have been loaded for the COCO dataset
    def __create_data_deque(self):
        self.img_deque = deque()
        for img in self.images:
            counter = 0
            captions = self.get_captions(img)
            while counter < self.caps_per_image:
                if len(self.get_captions(img)) > 0:
                    self.img_deque.append([img, captions[counter]])
                    counter += 1
                else:
                    break

    # takes as input the image file name, return list of captions
    def get_captions(self, img_file_name) -> list:
        return self.coco.get_captions(img_file_name)

    # image transforms
    def img_transforms(self, img):
        transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transformer(img)

    # function to display an image
    def display_image(self, idx) -> None:
        img = Image.open(os.path.join(self.images_path, self.img_deque[idx][0]))
        plt.imshow(img)
        plt.show()

    # load image as Image then transform it to tensor
    def load_img(self, idx):
        img_file_name = self.img_deque[idx][0]
        # convert the image to RGB to make sure all the images are 3D, because there are some images in grayscale
        img = Image.open(self.images_path + '/' + img_file_name).convert('RGB')
        if self.transform is None:
            img = self.img_transforms(img)
        else:
            img = self.transform(img)
        return img
    
    # method to return image as Image object
    def get_img(self, idx) -> Image:
        img_file_name = self.img_deque[idx][0]
        # convert the image to RGB to make sure all the images are 3D, because there are some images in grayscale
        img = Image.open(self.images_path + '/' + img_file_name).convert('RGB')
        return img

    # method to convert list of idx to caps
    def idx_to_caption(self, idx):
        cap = self.vocab.idx_to_caption(idx)
        cap = cap[1:-1]  # remove the <SOS> and <EOS>
        result = ''
        for word in cap:
            result += word + ' '
        return result.strip()

    # return the length of the current img deque
    def __len__(self):
        return len(self.img_deque)

    def __getitem__(self, idx):
        # get X: Image
        X = self.load_img(idx)
        # get y: Image Caption
        y = self.img_deque[idx][1]
        y = self.vocab.caption_to_idx(y)
        y = torch.tensor(y, dtype=torch.int64)
        return idx, X, y


class PAD:
    def __init__(self, pad_idx):
        """
        Collate to deal with the different lengths of captions
        This class make sure that in each batch, we take the biggest captions length and pad the others
        This is better than use a fixed sequence length for all the captions in the batch
        :param pad_idx: the index of <PAD> token
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        idxs = [item[0] for item in batch]
        imgs = [item[1].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[2] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return idxs, imgs, targets


# loader function to return the loader and the dataset interface
def get_loader(
        images_path,
        captions_path,
        freq_threshold,
        caps_per_image,
        mode='train',
        transform=None,
        batch_size=32,
        shuffle=True,
        idx2word=None, # not needed, just keeping for backward compatibility
        word2idx=None):

    if mode == 'test':
        assert batch_size == 1

    dataset_params = {
        'images_path': images_path,
        'captions_path': captions_path,
        'freq_threshold': freq_threshold,
        'caps_per_image': caps_per_image,
        'mode': mode,
        'transform': transform,
        'idx2word': idx2word,
        'word2idx': word2idx
    }
    dataset = MSCOCODataset(**dataset_params)

    pad_idx = dataset.vocab.word2idx['<PAD>']

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PAD(pad_idx=pad_idx)
    )

    return loader, dataset
