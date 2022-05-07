import os.path
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms

import json
import numpy as np
import time
import matplotlib.pyplot as plt

from PIL import Image
from nltk.translate import bleu_score


def load_vocab(word2idx_path,
               idx2word_path=None # leaving this for backwards compatibility but don't need it
               ):
    
    with open(word2idx_path) as json_file:
        string_to_index = json.load(json_file)
    
    if idx2word_path is None:
        idx_to_string = dict(zip(string_to_index.values(),
                                 string_to_index.keys()))
    
    else:
        # below is a less efficient way of loading the vocab and vocab may not exist
        print("it is prefered to just set word2idx_path and leave idx2word_path = None")
        with open(idx2word_path) as json_file:
            idx_to_string_json = json.load(json_file)

        idx_to_string = dict()
        for key in idx_to_string_json:
            idx_to_string[int(key)] = idx_to_string_json[key]

    return idx_to_string, string_to_index


# function to save the model
def save_model(epoch, encoder, decoder, training_loss, validation_loss, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'training_loss': training_loss,
        'validation_loss': validation_loss,
    }, os.path.join(checkpoint_path, 'model.pth'))


# function to load a checkpoint
def load_model(encoder, decoder, checkpoint_path):

    assert os.path.exists(checkpoint_path), "File not found"

    checkpoint = torch.load(os.path.join(checkpoint_path, 'model.pth'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    epoch = checkpoint['epoch']
    training_loss = checkpoint['training_loss']
    validation_loss = checkpoint['validation_loss']

    return epoch, encoder, decoder, training_loss, validation_loss


# function to train the model
def train(encoder, decoder, criterion, optimizer, train_loader, val_loader, total_epoch, device, checkpoint_path,
          print_every=1000, load_checkpoint=False):
    if load_checkpoint:
        e, encoder, decoder, training_loss, validation_loss = load_model(encoder, decoder, checkpoint_path)
    else:
        training_loss = []
        validation_loss = []
        e = 0

    encoder.to(device)
    decoder.to(device)

    file = open(os.path.join(checkpoint_path, "outputs.txt"), "a")
    for epoch in range(e, total_epoch):

        start_time = time.time()

        train_epoch_loss = 0
        val_epoch_loss = 0

        # Training phase
        encoder.train()
        decoder.train()

        for i, batch in enumerate(train_loader):
            idx, images, captions = batch
            images, captions = images.to(device), captions.to(device)

            # Zero the gradients.
            encoder.zero_grad()
            decoder.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions)

            loss = criterion(outputs.view(-1, decoder.vocab_size), captions.contiguous().view(-1))

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

            if i % print_every == 0:
                state = "Epoch: {:15} || Step: {:15} || Average Training Loss: {:.4f}".format('[{:d}/{:d}]'.format(epoch,total_epoch),
                                                                                              '[{:d}/{:d}]'.format(i,len(train_loader)),
                                                                                               train_epoch_loss / (i + 1))
                print(state)
                file.write(state+ '\n')
                file.flush()

        train_epoch_loss /= len(train_loader)
        training_loss.append(train_epoch_loss)

        # validation phase
        encoder.eval()
        decoder.eval()

        for i, batch in enumerate(val_loader):
            idx, images, captions = batch
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, decoder.vocab_size), captions.contiguous().view(-1))
            val_epoch_loss += loss.item()
            if i % print_every == 0:
                state = "Epoch: {:15} || Step: {:15} || Average Validation Loss: {:.4f}".format('[{:d}/{:d}]'.format(epoch,total_epoch),
                                                                                              '[{:d}/{:d}]'.format(i,len(val_loader)),
                                                                                               val_epoch_loss / (i + 1))
                print(state)
                file.write(state+'\n')
                file.flush()

        val_epoch_loss /= len(val_loader)
        validation_loss.append(val_epoch_loss)

        epoch_time = (time.time() - start_time) / 60 ** 1

        save_model(epoch, encoder, decoder, training_loss, validation_loss, checkpoint_path)
        state = "Epoch: [{0:d}/{1:d}] || Training Loss = {2:.2f} || Validation Loss: {3:.2f} || Time: {4:f}" \
                .format(epoch, total_epoch, train_epoch_loss, val_epoch_loss, epoch_time)
        file.write(100*"*" + '\n')
        file.write(state+ '\n')
        file.write(100*"*" + '\n')
        file.flush()
        print("*" * 100)
        print(state)
        print("*" * 100)

    file.close()

    return training_loss, validation_loss


# function to plot the training loss vs validation loss
def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend(loc='upper right')
    plt.title('Training vs Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


# function to predict a caption from an image                                            
def predict(encoder, decoder, image, idx2word, word2idx, device):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transformer(image)
    img = img.to(device)
    features = encoder(img.unsqueeze(0)).unsqueeze(1)
    outputs = decoder.predict(features, word2idx, 20)
    cap = [idx2word[word] for word in outputs]
    cap = cap[1:-1]
    result = ''
    for word in cap:
        result += word + ' '

    result += '.'
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # print(result.strip())
    return result.strip()

def plot_bleu_score_bar(b1_avg, b2_avg, b3_avg, b4_avg):
    x = np.array(["B1", "B2", "B3", "B4"])
    y = np.array([b1_avg, b2_avg, b3_avg, b4_avg])
    plt.bar(x, y)
    plt.xlabel('BLEU')
    plt.ylabel('SCORES')
    plt.title("BLEU scores")
    plt.show()


def save_params(save_path, batch_size, embed_size, hidden_size, num_layers, vocab_size):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    params = {
        'batch_size': batch_size,
        'embed_size': embed_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'vocab_size': vocab_size
    }
    with open(os.path.join(save_path, 'params.json'), "w") as outfile:
        json.dump(params, outfile)


def load_params(params_path):
    assert os.path.exists(params_path), "Folder not found"
    with open(os.path.join(params_path, 'params.json')) as json_file:
        params = json.load(json_file)
    return params


def load_model_losses(model_path):
    assert os.path.exists(model_path), "Folder not found"
    checkpoint = torch.load(os.path.join(model_path, 'model.pth'))
    training_loss = checkpoint['training_loss']
    validation_loss = checkpoint['validation_loss']
    return training_loss, validation_loss


def load_models_losses(model_paths):
    training_losses = []
    validation_losses = []
    for model_path in model_paths:
        training_loss, validation_loss = load_model_losses(model_path)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    return training_losses, validation_losses


def compare_models_losses(model_paths):
    training_losses, validation_losses = load_models_losses(model_paths)
    color_map = ['r', 'b', 'y', 'g']
    plt.figure()
    fig, axis = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(len(training_losses)):
        axis[0].plot(training_losses[i], color_map[i], label=f"Model {i + 1}")
        axis[1].plot(validation_losses[i], color_map[i], label=f"Model {i + 1}")

    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Loss')
    axis[0].set_title("Training Loss")
    axis[0].legend(loc="upper right")

    axis[1].set_xlabel('Loss')
    axis[1].set_ylabel('Epochs')
    axis[1].set_title("Validation Loss")
    axis[1].legend(loc="upper right")

    plt.show()


def comparing_bleu_scores(bleu_scores):
    x = ["B1", "B2", "B3", "B4"]
    x_axis = np.arange(len(x))
    plt.figure()
    plt.bar(x_axis - 0.1 * len(bleu_scores), bleu_scores[0], 0.2 * len(bleu_scores), label=f'Model 1')
    plt.bar(x_axis + 0.1 * len(bleu_scores), bleu_scores[1], 0.2 * len(bleu_scores), label=f'Model 2')

    plt.xticks(x_axis, x)
    plt.xlabel("Bleu Scores")
    plt.ylabel("Scores")
    plt.title("Comparing Bleu scores of different models")
    plt.legend(loc="upper right")
    plt.show()

def save_results(model_name: str,
                 checkpoint: Union[str, Path],
                 model_summary: str,
                 embed_size: int,
                 hidden_size: int,
                 num_layers: int,
                 vocab_size: int,
                 batch_size: int,
                 train_loss: list,
                 val_loss: list,
                 bleu_score: list):
    """Save results from each eval. 
    
    Bleu score takes a long time to calculate, so more efficient to do it and keep track or results in 
    a json file.
    
    Reading the file later will give a dictionary with keys equal to model names, and values equal to 
    results dictionary for thos models.
    """
    
    results_fname = Path('../model/summary_results')/'result_summary.json'
    
    results = {'model_name': model_name,
               'checkpoint': checkpoint,
               'embed_size': embed_size,
               'hidden_size': hidden_size,
               'LSTM_layers': num_layers,
               'vocab_size': vocab_size,
               'batch_size': batch_size,
               'model_summary': model_summary,
               'train_loss': train_loss,
               'val_loss': val_loss,
               'bleu_score': bleu_score
              }
    
    if results_fname.exists(): # create file if it doesn't exist
        # read file and add results dict. 
        with open(results_fname, 'r') as f:
            json_results = json.load(f)
            
        json_results[model_name] = results
        
    else:
        # create new master results dict and add results dict
        json_results = {model_name: results}
        
    with open(results_fname, 'w') as f:
        json.dump(json_results, f)

    
    
def evaluate_bleu_score(encoder, decoder, loader, dataset, device):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    capt_refs = []
    capt_hypos = []
    
    for i, batch in enumerate(loader):
        idx, image, caption = batch
        image = image.to(device)
        features = encoder(image).unsqueeze(1)
        outputs = decoder.predict(features, dataset.vocab.word2idx, 20)
        cap = [dataset.vocab.idx2word[word] for word in outputs]
        hypo = cap[1:-1]

        references = dataset.get_captions(dataset.img_deque[idx[0]][0])
        references = [ref.strip().lower().split() for ref in references]

        capt_refs.append(references)
        capt_hypos.append(hypo)

    b1 = bleu_score.corpus_bleu(capt_refs, capt_hypos, 
                                weights=(1.0, ))
    b2 = bleu_score.corpus_bleu(capt_refs, capt_hypos, 
                                weights=(0.5, 0.5))
    b3 = bleu_score.corpus_bleu(capt_refs, capt_hypos, 
                                weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    b4 = bleu_score.corpus_bleu(capt_refs, capt_hypos, 
                                weights=(0.25, 0.25, 0.25, 0.25))

    b1 = round(b1, 3)
    b2 = round(b2, 3)
    b3 = round(b3, 3)
    b4 = round(b4, 3) 

    return b1, b2, b3, b4