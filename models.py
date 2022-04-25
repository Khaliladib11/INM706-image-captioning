import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


# CNN Encoder
class Encoder(nn.Module):
    # constructor
    def __init__(self, embed_size, pretrained=True, model_weight_path=None):
        """
        Encoder is the first part to our model.
        The main porpose of encoder is to extract the usefule feature from an image
        We will use Resnet152 architecture pre-trained on ImageNet dataset
        Parameters
        ----------
        :param embed_size (int): the embed_size will be the output of the encoder since embed_size represents the input of the decoder
        :param pretrained (bool): if we want to load the pretrained weigth or not
        :param model_weight_path (sting): path to the pre trained weight
        """
        super(Encoder, self).__init__()
        # Load pretrained resnet152 on ImageNet
        if pretrained:
            self.resnet152 = models.resnet152(pretrained=True)
        else:
            self.resnet152 = models.resnet152(pretrained=False)
            self.resnet152.load_state_dict(torch.load(model_weight_path))

        # Freeze the parameters of pre trained model
        for param in self.resnet152.parameters():
            param.requires_grad_(False)

        # replace the last fully connected layer output with embed_size
        self.resnet152.fc = nn.Linear(in_features=self.resnet152.fc.in_features, out_features=1024)

        self.embed = nn.Linear(in_features=1024, out_features=embed_size)

        self.drop = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.drop(self.relu(self.resnet152(images)))
        features = self.embed(features)
        return features


# RNN Decoder
class Decoder(nn.Module):
    # constructor
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """
        Decoder is the second part of our model.
        Decoder takes as input the outputs of the encoder: the image feature vectors
        The input of the decoder and output of the encoder must be the same size
        Parameters
        ----------
        :param embed_size (int) : Dimensionality of image and word embeddings
        :param hidden_size (int) : number of features in hidden state of the RNN decoder
        :param vocab_size  (int) : The size of vocabulary or output size
        :param num_layers (int) : Number of layers

        """
        super(Decoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.dropout = nn.Dropout(0.2)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        # self.init_weights()

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        hiddens, c = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


    # get prediction
    def get_predict(self, features, max_length):
        idx = []
        inputs = features.unsqueeze(0)

        for i in range(max_length):
            hiddens, states = self.lstm(inputs)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            idx.append(predicted)
            inputs = self.embed(predicted)  # (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)

        idx = torch.stack(idx, 1)
        return idx