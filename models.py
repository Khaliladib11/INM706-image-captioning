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
        # self.bn = nn.BatchNorm1d(embed_size, momenturm=0.01) # could add this - haven't tested yet

    def forward(self, images):
        """Extract feature vectors from input images.
        images are size 
        
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1) 
            # returns tensor (batch_size, size1 x size2)
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
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.dropout = nn.Dropout(0.5)  # 0.2 originally, now 0.5
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        # self.model_init() # use zavier's initialization

        # To provide visibility, print tensor dimensions when instantiating the model
        # self.print_tensor_dimensions = True # flag to print if we need
        # self.init_weights()  # leaving this for now

    def forward(self, features, captions):
        
        ## Decoder forward useful parameters
        self.batch_size = features.shape[0]
        self.seq_length = captions.shape[1]

        # Initializing the hidden and cell states and 
        # flushing out previous hidden states
        # We don't want the previous batch to influence 
        # the output of next image-caption input. New on 1 May.
        self.hidden = self.init_hidden(self.batch_size)
        
        embeddings = self.embed(captions)
        # Alex (1 May): changing below from unsqueeze to 
        # view cos its clearer what we're doing
        features = features.view(self.batch_size, 1, self.embed_size)
        
        # embeddings is size (batch size, length of caption -1 
        #                     - remove EOS, embedding size)
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        lstm_out, hiddens = self.lstm(embeddings, self.hidden)
        lstm_out = self.dropout(lstm_out)
        
        # To chain multiple LSTM layers
        lstm_out = lstm_out.contiguous()
        
        lstm_out = lstm_out.view(self.batch_size,
                                 self.seq_length,
                                 self.hidden_size)  
        outputs = self.linear(lstm_out)
        
        return outputs
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes 
        # n_layers x batch_size x n_hidden,
        # initialized randomly, for hidden state 
        # and cell state of LSTM
                
        device = torch.device("cuda" if torch.cuda.is_available() 
                              else "cpu")
        # generates tuple of (hiden state, cell memory)
        hidden = (torch.randn(self.num_layers,
                              batch_size,
                              self.hidden_size).to(device),
                  torch.randn(self.num_layers,
                              batch_size,
                              self.hidden_size).to(device))
    
        return hidden

    # get the idxs of the predicted words.
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
