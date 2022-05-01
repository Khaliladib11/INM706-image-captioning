# import libraries
import matplotlib.pyplot as plt
import random
import itertools

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
                 word2idx=None
                 ):

        self.images_path = images_path
        self.captions_path = captions_path
        self.freq_threshold = freq_threshold
        self.caps_per_image = caps_per_image
        self.transform = transform

        self.coco = COCO(self.images_path, self.captions_path)
        # self.images = self.coco.images
        self.images = self.coco.images
        random.seed(706)

        if mode == 'train':
            if word2idx is None:
                # imgs = deque(random.sample(self.coco.images[:80_000], 50_000))
                imgs = deque(random.sample(self.images[:80_000], 50_000))
                caps = []
                for img in imgs:
                    for cp in self.coco.imgs_caps_dict[img]:
                        caps.append(cp)

                self.vocab = Vocabulary(freq_threshold=self.freq_threshold)
                self.vocab.build_vocabulary(caps)
            else:
                idx2word = dict(zip(word2idx.values(), word2idx.keys()))
                self.vocab = Vocabulary(freq_threshold=self.freq_threshold, idx2word=idx2word, word2idx=word2idx)
        
        self.word2idx = word2idx
        self.idx2word = idx2word

        """
        self.coco = COCO(self.images_path, self.captions_path)
        if idx2word is None and word2idx is None:
            self.vocab = Vocabulary(freq_threshold=self.freq_threshold)
            self.vocab.build_vocabulary(self.coco.captions_to_list())
        else:
            self.vocab = Vocabulary(freq_threshold=self.freq_threshold, idx2word=idx2word, word2idx=word2idx)
        """
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

    def get_captions(self, img_file_name):
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

    # load image as Image then transform it to tensor
    def load_img(self, idx):
        img_file_name = self.img_deque[idx][0]
        # convert the image to RGB to make sure all the images are 3D, because there are some images in grayscale
        img = Image.open(self.images_path/img_file_name).convert('RGB')
        if self.transform is None:
            img = self.img_transforms(img)
        else:
            img = self.transform(img)
        return img

    def idx_to_caption(self, idx):
        cap = ''
        for token in self.vocab.idx_to_caption(idx):
            cap += token + ' '
        return cap

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
        self.pad_idx = pad_idx

    def __call__(self, batch):
        idxs = [item[0] for item in batch]
        imgs = [item[1].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[2] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return idxs, imgs, targets


def get_loader(
        images_path,
        captions_path,
        freq_threshold,
        caps_per_image,
        mode='train',
        transform=None,
        batch_size=32,
        shuffle=True,
        word2idx=None):
    
    dataset_params = {
        'images_path': images_path,
        'captions_path': captions_path,
        'freq_threshold': freq_threshold,
        'caps_per_image': caps_per_image,
        'mode': mode,
        'transform': transform,
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
