from vocabulary import Vocabulary
import json
from pathlib import Path
import numpy as np


def build_vocab(freq_threshold=5, 
                captions_file='captions_train2017.json',
                vocab_save_name=''):
    """build a vocabulary using any captions json file we want.
    This enables us to build vocab independent of the test set we are loading in for training for examle.
    
    freq_threshold: integer. Only words occuring equal to or more than the threshold are included in vocab.
    vocab_save_name: string. Root for word2idx.json file which we save.

    When building vocab with full captions_train2017.json file, we found that adjusting the freq threshold
    gave vocabs of the following size:
    
        With FREQ_THRESHOLD = 1, vocab size is 26852
        With FREQ_THRESHOLD = 2, vocab size is 16232
        With FREQ_THRESHOLD = 3, vocab size is 13139
        With FREQ_THRESHOLD = 4, vocab size is 11360
        With FREQ_THRESHOLD = 5, vocab size is 10192
        With FREQ_THRESHOLD = 6, vocab size is 9338
        
    Setting freq_threshold at 2 is probably fine, but setting higher has the above effect on vocab size.
    
    This function returns none but saves the vocab.idx and vocab.word2idx attributes
    as json files with name idx2word.json and word2idx.json in the vocabulary folder
    """
    anns_path = Path('../Datasets/coco/annotations/')
    vocab_path = Path('../vocabulary/')

    if isinstance(captions_file, list):
        annotations = []
        for file in captions_file:
            with open(anns_path/file, 'r') as f:
                annotations.extend(json.load(f)['annotations'])
    else:
        with open(anns_path/captions_file, 'r') as f:
            annotations = json.load(f)['annotations']

    print(f"There are {len(annotations)} captions in the data set")
    
    vocab = Vocabulary(freq_threshold, 
                       # sequence_length
                      )
    captions = []
    for d in annotations:
        captions.append(d['caption'])
    vocab.build_vocabulary(captions)

    print("With FREQ_THRESHOLD = {}, vocab size is {}"
          .format(freq_threshold, len(vocab.idx2word)))
    
    with open(vocab_path/f'{vocab_save_name}word2idx.json', 'w') as f:
        json.dump(vocab.word2idx, f)
        
    return None

def prepare_datasets(train_percent = 0.87, super_categories=None,
                    max_train=15000, max_val=3000, max_test=3000,
                    save_name=None, random_seed=42):
    """Prepare train/val/test datasets for training.
    Can reduce size of data by only picking certain super_categories - e.g. sports. This will 
    select only images that contain listed super_categories.
    
    train_percent, float. between 0 and 1. How to split eligible images between train and val.
    super_categories, list. If None then selects all data
    save_name, string or None. If string then files will be saved as [save_name]_captions_train.json etc

    This function is a bit messy as it was converted from a Jupyter Notebook    
    """
    
    annotations_folder = Path(r'../Datasets/coco/annotations')
    image_folder = Path(r'../Datasets/coco/images/train2017')
    image_folder_test = Path(r'../Datasets/coco/images/val2017')
    
    STAGE = 'train'
    TRAIN_PCT = train_percent
    # instances file contains meta data of images including which categories of object are in each image
    instances_file = annotations_folder/f'instances_{STAGE}2017.json'
    captions_file = annotations_folder/f'captions_{STAGE}2017.json'

    with open(instances_file) as f:
        instances = json.load(f)
    with open(captions_file) as f:
        captions = json.load(f)
        
    # build list of categories we will use
    if super_categories:
        supers = super_categories
    else:
        supers = []
        for cat in instances['categories']:
            supers.append(cat['supercategory'])
        supers = list(set(supers))
        
    
    ids = [] # build list of categories that come under our super categories
    id_dict = {} # build dict with image ids as keys and name as values.

    for d in instances['categories']:
        if d['supercategory'] in supers:
            ids.append(d['id'])
            id_dict[d['id']] = d['name']

    
    img_list = [] # images we will choose for our data set
    full_img_list = [int(f.stem) for f in image_folder.glob('**/*')] # all images in the train2017 set
    for d in instances['annotations']:
        if d['category_id'] in ids:
            img_list.append(d['image_id'])
            # annotations.append(d)
    img_list = list(set(img_list)) # removes repeated images
    
    # randomly sample images for our training set
    rng = np.random.default_rng(random_seed)
    imgs_train = rng.choice(img_list,
                            size =int(TRAIN_PCT * len(img_list)),
                            replace=False)
    # randomly sample from remaining images for our val set
    imgs_val = list(set(img_list).difference(set(imgs_train)))

    # reduce size of data if we have set smaller max sizes
    if len(imgs_train > max_train):
        imgs_train = imgs_train[:max_train]
    if len(imgs_val) > max_val:
        imgs_val = imgs_val[:max_val]

    # rebuild ditionaries in same format as captions2017.json just with our selected data
    captions_full = dict(zip(full_img_list, 
                             [[] for _ in range(len(full_img_list))]))
    images_dict = dict(zip(full_img_list, 
                             [[] for _ in range(len(full_img_list))]))

    for d in captions['annotations']:
        captions_full[d['image_id']].append(d)

    for d in captions['images']:
        images_dict[d['id']].append(d)

    def create_lists(img_list, capts_dict, imgs_dict):
        new_capt_list = []
        new_img_list = []
        for img_id in img_list:
            new_capt_list.extend(capts_dict[img_id])
            new_img_list.extend(imgs_dict[img_id])
        return new_capt_list, new_img_list

    n_capt_list_train, n_img_list_train = create_lists(imgs_train,
                                                       captions_full,
                                                       images_dict)
    n_capt_list_val, n_img_list_val = create_lists(imgs_val,
                                                   captions_full,
                                                   images_dict)

    new_captions_train = {'info': captions['info'],
                        'images': n_img_list_train,
                        'annotations': n_capt_list_train}

    new_captions_val = {'info': captions['info'],
                        'images': n_img_list_val,
                        'annotations': n_capt_list_val}

    train_img_paths = {'image_paths': [image_folder/(str(id).zfill(12) + '.jpg')
                                       for id in imgs_train]}

    val_img_paths = {'image_paths': [image_folder/(str(id).zfill(12) + '.jpg')
                                       for id in imgs_val]}
    
    
    # The code below repeats what was done above. It is an inefficient implementation as it could ahve been
    # wrapped as a function and called twice. TODO: refactor code to make more concise.
    
    STAGE = 'val' # we are using coco validation set for our test set
    instances_file = annotations_folder/f'instances_{STAGE}2017.json'
    captions_file = annotations_folder/f'captions_{STAGE}2017.json'

    with open(instances_file) as f:
        instances_test = json.load(f)
    with open(captions_file) as f:
        captions_test = json.load(f)

    img_list_test = []
    full_img_list_test = [int(f.stem) for f in image_folder_test.glob('**/*')]
    annotations_test = []
    for d in instances_test['annotations']:
        if d['category_id'] in ids:
            img_list_test.append(d['image_id'])
            annotations_test.append(d)
    img_list_test = list(set(img_list_test))
    if len(img_list_test) > max_test:
        img_list_test = img_list_test[:max_test]

    captions_full_test = dict(zip(full_img_list_test, 
                             [[] for _ in range(len(full_img_list_test))]))
    images_dict_test = dict(zip(full_img_list_test, 
                             [[] for _ in range(len(full_img_list_test))]))

    for d in captions_test['annotations']:
        captions_full_test[d['image_id']].append(d)

    for d in captions_test['images']:
        images_dict_test[d['id']].append(d)

    n_capt_list_test, n_img_list_test = create_lists(img_list_test,
                                                     captions_full_test,
                                                     images_dict_test)

    new_captions_test = {'info': captions_test['info'],
                        'images': n_img_list_test,
                        'annotations': n_capt_list_test}

    test_img_paths = {'image_paths': [image_folder_test/(str(id).zfill(12) + '.jpg')
                                       for id in img_list_test]}
    
    if save_name:
        name = save_name + '_captions_'
    else:
        name = 'custom_captions_'
    save_files = {f'{name}train': new_captions_train,
                  f'{name}val': new_captions_val,
                  f'{name}test': new_captions_test,
                 }

    # Save files to new json files
    for key, val in save_files.items():
        with open(annotations_folder/f'{key}.json', 'w') as json_file:
            json.dump(val, json_file)
            
    print("train dataset has {} images\n val dataset has {} images\n test dataset has {} images".format(len(imgs_train), len(imgs_val), len(img_list_test)))
