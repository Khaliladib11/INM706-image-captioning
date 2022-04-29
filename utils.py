from Vocabulary import Vocabulary
import json
from pathlib import Path

def build_vocab(freq_threshold=2, sequence_length=40, captions_file='captions_train2017.json'):
    """build a vocabulary using any captions json file we want.
    This enables us to build vocab independent of the test set we are loading in for training for examle.
    
    freq_threshold: integer. Only words occuring equal to or more than the threshold are included in vocab.
    sequence_length: truncate captions at number of words = sequence length for building vocab. We should set
                     this as a high number - 40 is fine.
                     
    When building vocab with full captions_train2017.json file, we found that adjusting the freq threshold
    gave vocabs of the following size:
    
        With FREQ_THRESHOLD = 1, vocab size is 26852
        With FREQ_THRESHOLD = 2, vocab size is 16232
        With FREQ_THRESHOLD = 3, vocab size is 13139
        With FREQ_THRESHOLD = 4, vocab size is 11360
        With FREQ_THRESHOLD = 5, vocab size is 10192
        With FREQ_THRESHOLD = 6, vocab size is 9338
        
    Setting freq_threshold at 2 is probably fine, but setting higher has the above effect on vocab size.
    
    This function returns none but saves the vocab.idx_to_string and vocab.string_to_idx attributes
    as json files with name idx_to_string.json and string_to_index.json in the vocabulary folder
    """
    anns_path = Path('Datasets/coco/annotations/')
    vocab_path = Path('vocabulary/')

    with open(anns_path/captions_file, 'r') as f:
        annotations = json.load(f)
# 
    print(f"There are {len(annotations['annotations'])} captions in the data set")
    
    vocab = Vocabulary(freq_threshold, sequence_length)
    captions = []
    for d in annotations['annotations']:
        captions.append(d['caption'])
    vocab.build_vocabulary(captions)

    print("With FREQ_THRESHOLD = {}, vocab size is {}"
          .format(freq_threshold, len(vocab.idx_to_string)))
    
    with open(vocab_path/'idx_to_string.json', 'w') as f:
        json.dump(vocab.idx_to_string, f)
    with open(vocab_path/'string_to_index.json', 'w') as f:
        json.dump(vocab.string_to_index, f)
        
    return None

def prepare_datasets(train_percent = 0.87, super_categories=None,
                    max_train=15000, max_val=3000, max_test=3000):
    """Prepare train/val/test datasets for training.
    Can reduce size of data by only picking certain super_categories - e.g. sports
    
    train_percent, float. between 0 and 1. How to split eligible images between train and val.
    super_categories, list. If None then selects all data
    """
    
    annotations_folder = Path(r'Datasets/coco/annotations')
    image_folder = Path(r'Datasets/coco/images/train2017')
    image_folder_test = Path(r'Datasets/coco/images/val2017')
    
    STAGE = 'train'
    TRAIN_PCT = train_percent
    instances_file = annotations_folder/f'instances_{STAGE}2017.json'
    captions_file = annotations_folder/f'captions_{STAGE}2017.json'

    with open(instances_file) as f:
        instances = json.load(f)
    with open(captions_file) as f:
        captions = json.load(f)
        
    if super_categories:
        supers = super_categories
    else:
        supers = []
        for cat in instances['categories']:
            supers.append(cat['supercategory'])
        supers = list(set(supers))
        
    ids = []
    id_dict = {}
    id_to_name = dict(zip([cat['id'] for cat in instances['categories']],
                         [cat['name'] for cat in instances['categories']]))
    for d in instances['categories']:
        if d['supercategory'] in supers:
            ids.append(d['id'])
            id_dict[d['id']] = d['name']

    img_list = []
    full_img_list = [int(f.stem) for f in image_folder.glob('**/*')]
    annotations = []
    for d in instances['annotations']:
        # full_img_list.append(d['image_id'])
        if d['category_id'] in ids:
            img_list.append(d['image_id'])
            annotations.append(d)
    img_list = list(set(img_list))
    
    rng = np.random.default_rng(42)
    imgs_train = rng.choice(img_list,
                            size =int(TRAIN_PCT * len(img_list)),
                            replace=False)
    imgs_val = list(set(img_list).difference(set(imgs_train)))

    if len(imgs_train > max_train):
        imgs_train = imgs_train[:max_train]
    if len(imgs_val) > max_val:
        imgs_val = imgs_val[:max_val]

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
        # full_img_list.append(d['image_id'])
        if d['category_id'] in ids:
            img_list_test.append(d['image_id'])
            annotations_test.append(d)
    img_list_test = list(set(img_list_test))
    if img_list_test > max_test:
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
    
    save_files = {'sports_captions_train': new_captions_train,
                  'sports_captions_val': new_captions_val,
                  'sports_captions_test': new_captions_test,
                 }

    for key, val in save_files.items():
        with open(annotations_folder/f'{key}.json', 'w') as json_file:
            json.dump(val, json_file)