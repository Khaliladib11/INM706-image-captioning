import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Vocabulary:

    # Constructor
    def __init__(self, freq_threshold, sequence_length=6, idx_to_string=None, string_to_index=None):
        """
        Constructor to create vocabulary of words and to tokenize them.
        :param freq_threshold (int): if a word is not repeated enough don't add it to the dictionary
        :param sequence_length (int): all sequences must be the same lenght, so this param is used to pad or cut from sentence
        """
        if idx_to_string is None or string_to_index is None:
            self.idx_to_string = {
                0: '<PAD>',  # to pad all the captions to be the same size
                1: '<SOS>',  # Start of sentence
                2: '<EOS>',  # End of sentence
                3: '<UNK>',  # Unknown Token
            }
            self.string_to_index = {
                '<PAD>': 0,
                '<SOS>': 1,
                '<EOS>': 2,
                '<UNK>': 3,
            }
        
        else:
            self.idx_to_string = idx_to_string
            self.string_to_index = string_to_index
        
        self.freq_threshold = freq_threshold
        self.sequence_length = sequence_length
        self.stop_words = stopwords.words('english')  # stop words in english


    # return the length of the vocabulary
    def __len__(self):
        return len(self.idx_to_string)

    # tokenize a pieace of text
    def tokenizer_eng(self, sentence):
        sentence = re.sub('\W+', ' ', sentence.lower())  # Remove all special characters, punctuation and spaces
        tokenized_sentence = word_tokenize(sentence)  # Tokenize the words
        #tokenized_sentence = [w for w in tokenized_sentence if w not in self.stop_words]  # remove stop word

        return tokenized_sentence

    # medthod to build the vocabulary for us
    def build_vocabulary(self, sentences_list):
        frequencies = {}
        idx = len(self.idx_to_string)
        for sentence in sentences_list:
            for token in self.tokenizer_eng(sentence):
                if token not in self.string_to_index:
                    frequencies[token] = 1
                    self.string_to_index[token] = idx
                    self.idx_to_string[idx] = token
                    idx += 1
                else:
                    frequencies[token] += 1

        for word in frequencies:
            if frequencies[word] < self.freq_threshold:
                idx = self.string_to_index[word]
                self.string_to_index.pop(word)
                self.idx_to_string.pop(idx)

    # method to convert words to numerical values
    def numericalize(self, sentence):
        tokenized_sentence = self.tokenizer_eng(sentence)  # First tokenize the sentence
        # then convert the words to numerical idxs from our vocab
        idx_of_sentence = [self.string_to_index[word] if word in self.string_to_index else self.string_to_index['<UNK>']
                           for word in tokenized_sentence]

        if len(idx_of_sentence) >= self.sequence_length:
            idx_of_sentence = idx_of_sentence[:self.sequence_length]
        else:
            idx_of_sentence.extend([self.string_to_index['<PAD>']] * (self.sequence_length - len(idx_of_sentence)))
        return idx_of_sentence

    # method to convert idxs to words
    def convert_to_text(self, idxs):
        # convert idxs to words using idx_to_string
        sentence = [self.idx_to_string[idx] if idx in self.idx_to_string else '' for idx in idxs]
        # remove <PAD> if there is any
        sentence = list(filter(lambda word: word != '<PAD>', sentence))
        return sentence
