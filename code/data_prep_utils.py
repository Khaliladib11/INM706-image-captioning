from vocabulary import Vocabulary
import json
from pathlib import Path
import numpy as np

def build_vocab(freq_threshold=2,
                captions_file='captions_train2017.json'):
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
# 
    print(f"There are {len(annotations)} captions in the data set")
    
    vocab = Vocabulary(freq_threshold)
    captions = []
    for d in annotations:
        captions.append(d['caption'])
    vocab.build_vocabulary(captions)

    print("With FREQ_THRESHOLD = {}, vocab size is {}"
          .format(freq_threshold, len(vocab.idx2word)))
    
    with open(vocab_path/'string_to_index.json', 'w') as f:
        json.dump(vocab.word2idx, f)
        
    return None