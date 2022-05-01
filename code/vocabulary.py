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
        """

        self._index = 0  # index to keep track of the last added id in the dicts
        if idx2word is None and word2idx is None:
            self.idx2word = {}  # dict to convert idx to word
            self.word2idx = {}  # dict to convert word to idx

        else:
            self.idx2word = idx2word
            self.word2idx = word2idx
            for idx in idx2word:
                if idx >= self._index:
                    self._index += 1

        self.freq_threshold = freq_threshold  # threshold
        self.stop_words = stopwords.words('english')  # stopping words in english

    # tokenize a piece of text
    @staticmethod
    def tokenizer_eng(sentence):
        sentence = re.sub('\W+', ' ', sentence.lower())  # Remove all special characters, punctuation and spaces
        tokenized_sentence = word_tokenize(sentence)  # Tokenize the words
        return tokenized_sentence

    # add new word to the dicts
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self._index
            self.idx2word[self._index] = word
            self._index += 1

    # method to build the vocabulary
    def build_vocabulary(self, captions_deque):
        counter = Counter()
        tokens = []
        for caption in captions_deque:
            tokens.extend(self.tokenizer_eng(caption))

        counter.update(tokens)
        words = [word for word, count in counter.items() if count >= self.freq_threshold]

        self.add_word('<SOS>')
        self.add_word('<EOS>')
        self.add_word('<PAD>')
        self.add_word('<UNK>')

        for word in words:
            self.add_word(word)

    # get a word from idx
    def get_word(self, idx):
        if idx in self.idx2word.keys():
            return self.idx2word[idx]
        else:
            # return UNK
            return self.idx2word[3]

    # get an idx from word
    def get_idx(self, word):
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            # return idx of UNK
            return self.word2idx['<UNK>']

    # convert a caption to idxs
    def caption_to_idx(self, sentence):
        tokens = self.tokenizer_eng(sentence)
        tokens.insert(0, '<SOS>')
        tokens.append('<EOS>')
        idxs = [self.get_idx(word) for word in tokens]
        return idxs

    # convert idxs to caption
    def idx_to_caption(self, idxs):
        caption = [self.get_word(idx) for idx in idxs]
        caption = list(filter(lambda word: word != '<PAD>', caption))
        return caption

    def export_vocab(self, save_path):
        with open(os.path.join(save_path, 'idx2word.json'), "w") as outfile:
            json.dump(self.idx2word, outfile)

        with open(os.path.join(save_path, 'word2idx.json'), "w") as outfile:
            json.dump(self.word2idx, outfile)

    # return the length of the vocab
    def __len__(self):
        return len(self.idx2word)
