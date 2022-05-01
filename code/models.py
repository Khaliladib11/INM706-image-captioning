import torch
import torch.nn as nn
import torchvision.models as models


# CNN Encoder
class Encoder(nn.Module):
    # constructor
    def __init__(self, embed_size, pretrained=True, model_weight_path=None):
        """
        Encoder is the first part to our model.
        The main purpose of encoder is to extract the useful feature from an image
        We will use Resnet152 architecture pre-trained on ImageNet dataset
        Parameters
        ----------
        :param embed_size (int): the embed_size will be the output of the encoder since embed_size represents the input of the decoder
        :param pretrained (bool): if we want to load the pre-trained weight or not
        :param model_weight_path (sting): path to the pre-trained weight
        """
        super(Encoder, self).__init__()
        # Load pretrained resnet152 on ImageNet
        if pretrained:
            resnet = models.resnet152(pretrained=True)
        else:
            resnet = models.resnet152(pretrained=False)
            resnet.load_state_dict(torch.load(model_weight_path))

        # Freeze the parameters of pre-trained model
        for param in resnet.parameters():
            param.requires_grad_(False)

        # replace the last fully connected layer output with embed_size
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
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

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.init_weights()

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=self.device), \
                torch.zeros((1, batch_size, self.hidden_size), device=self.device))

    def forward(self, features, captions):
        """
        embeddings = self.dropout(self.embed(captions))
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
        """
        captions = captions[:, :-1]
        batch_size = features.shape[0]  # features is of shape (batch_size, embed_size)
        self.hidden = self.init_hidden(batch_size)
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        outputs = self.linear(lstm_out)
        return outputs

    # greedy search
    def predict(self, features, max_length, idx2word):
        caption = []
        inputs = features.unsqueeze(0)

        for i in range(max_length):
            hiddens, states = self.lstm(inputs)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            caption.append(predicted.item())
            inputs = self.embed(predicted)  # (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)

            if idx2word[predicted.item()] == "<EOS>":
                break

        return caption
