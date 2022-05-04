import os
import re
import numpy as np
import json
import pickle
from collections import deque, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Vocabulary:

    def __init__(self, freq_threshold, idx2word=None, word2idx=None):
        """
        Constractor for vocabulary class to build vocabulary
        :param freq_threshold: if a word is not repeated enough don't add it to the dictionary
        :param idx2word: ready idx2word dict (optional)
        :param word2idx: read word2idx dict (optimal)
        """
        
        self._index = 0  # index to keep track of the last added id in the dicts
        
        # if idx2word and word2idx are None, initial them as empty dict
        if idx2word is None and word2idx is None:
            self.idx2word = {}  # dict to convert idx to word
            self.word2idx = {}  # dict to convert word to idx
        
        # else load them and take the max key and add 1 to it
        else:
            assert len(idx2word) == len(word2idx), "length of idx2word and word2idx are not equal"
            self.idx2word = idx2word
            self.word2idx = word2idx
            for idx in idx2word:
                if idx >= self._index:
                    self._index += 1

        self.freq_threshold = freq_threshold  # threshold
        self.stop_words = stopwords.words('english')  # stopping words in english

     # tokenize a piece of text, takes as input a string and return tokenized list
    @staticmethod
    def tokenizer_eng(sentence) -> list:
        sentence = re.sub('\W+', ' ', sentence.lower())  # Remove all special characters, punctuation and spaces
        tokenized_sentence = word_tokenize(sentence)  # Tokenize the words
        return tokenized_sentence

    # add new word to the dicts
    def add_word(self, word) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = self._index
            self.idx2word[self._index] = word
            self._index += 1

    # method to build the vocabulary
    def build_vocabulary(self, captions_deque) -> None:
        counter = Counter()
        tokens = []
        for caption in captions_deque:
            tokens.extend(self.tokenizer_eng(caption))

        counter.update(tokens)
        words = [word for word, count in counter.items() if count >= self.freq_threshold]

        # add some special tokens
        self.add_word('<SOS>')  # start of sentence
        self.add_word('<EOS>')  # end of sentence
        self.add_word('<PAD>')  # padding
        self.add_word('<UNK>')  # unknown token

        # add all the word in words list to the dicts
        for word in words:
            self.add_word(word)

    # get a word from idx
    def get_word(self, idx) -> str:
        # check if the index is existed as key in idx2word dict, if not, return <UNK> token
        if idx in self.idx2word.keys():
            return self.idx2word[idx]
        else:
            # return UNK
            return '<UNK>'

    # get an idx from word
    def get_idx(self, word):
        # check if the word is existed as key in word2idx dict, if not, return the index of <UNK> token
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            # return idx of UNK
            return self.word2idx['<UNK>']

    # convert a caption to list of indexes, takes as input a sentence, return a list of indexes
    def caption_to_idx(self, sentence) -> list:
        tokens = self.tokenizer_eng(sentence)  # first tokenize the sentence
        tokens.insert(0, '<SOS>')  # insert <SOS> in the beginning of the tokens list
        tokens.append('<EOS>')  # append <EOS> in the end of the tokens list
        indexes = [self.get_idx(word) for word in tokens]  # convert each token to index
        return indexes

    # convert indexes to caption, takes as input list of indexes, return list of tokens
    def idx_to_caption(self, indexes) -> list:
        caption = [self.get_word(idx) for idx in indexes]   # first convert the indexes to tokens
        caption = list(filter(lambda word: word != '<PAD>', caption))  # filtering the <PAD> tokens
        return caption
    
    # method to export the vocab dict as json file, takes as input the path where we want to save the file
    def export_vocab(self, save_path):
        with open(os.path.join(save_path, 'idx2word.json'), "w") as outfile:
            json.dump(self.idx2word, outfile)

        with open(os.path.join(save_path, 'word2idx.json'), "w") as outfile:
            json.dump(self.word2idx, outfile)

    # return the length of the vocab
    def __len__(self):
        return len(self.idx2word)
