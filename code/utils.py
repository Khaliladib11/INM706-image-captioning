import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms

import json
import numpy as np
import time
import matplotlib.pyplot as plt


def save_model(epoch, encoder, decoder, training_loss, validation_loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'hyper_params': hyper_params
    }, checkpoint_path)


def load_model(encoder, decoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    training_loss = checkpoint['training_loss']
    validation_loss = checkpoint['validation_loss']
    hyper_params = checkpoint['hyper_params']

    return encoder, decoder, training_loss, validation_loss


def train(encoder, decoder, 
          criterion, optimizer,
          train_loader, val_loader,
          total_epoch, device,
          checkpoint_path,
          training_loss=None, validation_loss=None,
          print_every=1000):
    
    encoder.to(device)
    decoder.to(device)

    if not training_loss:
        training_loss = []
    if not validation_loss:
        validation_loss = []

    for epoch in range(total_epoch):

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
                print("Step: {:15}|| Average Training Loss: {:.4f}".
                      format('[{:d}/{:d}]'.format({0:d}/{1:d}),
                             train_epoch_loss / (id + 1)))

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
                print("Step: [{0:d}/{1:d}] || Average Validation Loss: {2:.4f}".format(i, len(val_loader),
                                                                                       val_epoch_loss / (i + 1)))

        val_epoch_loss /= len(val_loader)
        validation_loss.append(val_epoch_loss)

        epoch_time = (time.time() - start_time) / 60 ** 1

        save_model(epoch, encoder, decoder, training_loss, validation_loss, checkpoint_path)
        print("######################################################################")
        print(
            "Epoch: [{0:d}/{1:d}] || Training Loss = {2:.2f} || Training Perplexity: {3:.2f} || Validation Loss: {4:.2f} || Validation Perplexity: {5:.2f}|| Time: {5:f}" \
                .format(epoch, total_epoch, train_epoch_loss, np.exp(train_epoch_loss), val_epoch_loss,
                        np.exp(val_epoch_loss), epoch_time))
        print("######################################################################")

    return training_loss, validation_loss


def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend(loc='upper right')
    plt.show()