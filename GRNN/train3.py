import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
import datetime
from torch.utils.data import DataLoader, sampler
import random

# from utils import split_data, CustomDataloader
from model import DocSenModel
# from preprocessing import GRNNDataset
from prep3 import GRNNDataset, CustomDataloader


def split_data(dataset, train_split, val_split, test_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_train = int(np.floor(train_split * dataset_size))
    split_val = int(np.floor(val_split * dataset_size))
    return indices[:split_train], indices[split_train:split_train + split_val], indices[split_train + split_val:]


def train(dataset, sentence_model, gnn_type, gnn_output):
    train_indices, val_indices, test_indices = split_data(dataset, 0.8, 0.1, 0.1)

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    dataloader_train = CustomDataloader(dataset, train_sampler, batch_size=64)
    dataloader_valid = CustomDataloader(dataset, valid_sampler, batch_size=64)

    # Model parameter
    if sentence_model == 0:
        print("Sentence model: convolution")
        sentence_model = DocSenModel.SentenceModel.CONV
    else:
        print("Sentence model: LSTM")
        sentence_model = DocSenModel.SentenceModel.LSTM

    if gnn_type == 0:
        print("GNN type: forward")
        gnn_type = DocSenModel.GnnType.FORWARD
    else:
        print("GNN type: forward-backward")
        gnn_type = DocSenModel.GnnType.FORWARD_BACKWARD

    if gnn_output == 0:
        print("GNN output: last")
        gnn_output = DocSenModel.GnnOutput.LAST
    else:
        print("GNN output: avg")
        gnn_output = DocSenModel.GnnOutput.AVG

    # Model
    model = DocSenModel(dataset.n_classes, sentence_model, gnn_output, gnn_type, dataset.embedding, cuda=True)

    # Training parameters
    lr = 0.03
    l2_reg = 1e-5
    epochs = 70

    criterion = nn.CrossEntropyLoss()  # loss_function
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg)

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    t0 = time.time()
    for epoch in range(0, epochs):
        print("\n ======================= Epoch {:}/{:} =====================".format(epoch + 1, epochs))
        start = time.time()
        '''
        EARLY STOPPING
        if early_stopping > 0:
            if epoch - min_loss_epoch >= early_stopping:
                print(f"No training improvement over the last {early_stopping} epochs. Aborting.")
                break
        '''
        for step, batch in enumerate(dataloader_train):
            # Forward pass for each single document in the batch
            predictions = None
            labels = None
            matches = 0
            for (doc, label) in batch:
                if len(doc) == 0:
                    continue
                try:
                    prediction = model(doc)
                    prediction = prediction.unsqueeze(0)
                    predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                    label = torch.Tensor([label])
                    label = label.long().to(model._device)
                    labels = label if labels is None else torch.cat((labels, label))

                    if label == torch.argmax(prediction):
                        matches += 1
                except (AttributeError, RuntimeError) as e:
                    print("Some error occurred. Ignoring this document. Error:")
                    print(e)
                    continue

            # Compute the loss
            loss_object = criterion(predictions, labels)
            loss = loss_object.item()
            train_loss.append(loss)

            accuracy = float(matches) / float(len(predictions))
            train_acc.append(accuracy)

            # Reset the gradients in the optimizer.
            # Otherwise past computations would influence new computations.
            optimizer.zero_grad()
            loss_object.backward()
            optimizer.step()

            if step % 1000 == 0:
                print("Batch: [{:>5}/{:>5}]\t"
                      "Batch Time {}\t"
                      "Loss: {}, Accuracy: {}".format(step + 1, len(dataloader_train),
                                                      str(datetime.timedelta(
                                                          seconds=int(round(time.time() - start)))),
                                                      loss, accuracy))

            # VALIDATION
                        # Test on a single batch from the validation set
            # Set model to evaluation mode
            model.eval()
            batch_valid_indices = random.choices(val_indices, k=64)
            predictions = None
            labels = None
            matches = 0
            for (doc, label) in dataloader_valid._batch_iterator(batch_valid_indices):
                if len(doc) == 0:
                    continue
                try:
                    prediction = model(doc)
                    prediction = prediction.unsqueeze(0)
                    predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                    label = torch.Tensor([label])
                    label = label.long().to(model._device)
                    labels = label if labels is None else torch.cat((labels, label))

                    if label == torch.argmax(prediction):
                        matches += 1
                except (AttributeError, RuntimeError) as e:
                    print("Some error occurred. Ignoring this document. Error:")
                    print(e)
                    continue

            # Compute the loss
            loss_object = criterion(predictions, labels)
            valid_loss.append(loss_object.item())

            accuracy = float(matches) / float(len(predictions))
            valid_acc.append(accuracy)

            # Set model back to training mode
            model.train()

            if step % 10 == 0:
                print(f"  Epoch {epoch+1:>2} of {epochs} - Batch {step+1:>5} of {len(dataloader_train):>5}  -"
                      f"   Tr.-Loss: {train_loss[-1]:.4f}   Val.-Loss: {valid_loss[-1]:.4f}"
                      f"   Tr.-Acc.: {train_acc[-1]:.2f}   Val.-Acc.: {valid_acc[-1]:.2f}")

        print("Saving training progress checkpoint...")
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, test_indices)

        # Update Learning Rate
        #learning_rate = 0.8 * learning_rate
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        adjust_learning_rate(optimizer, 0.8)

    print("END TRAINING\n")
    print("Total training time {:} (h:mm:ss)".format(
        str(datetime.timedelta(seconds=int(round(time.time() - t0))))))


def save_checkpoint(epoch, model, optimizer, test_indices):  # , word_map):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'test_indices': test_indices}
    # ,'word_map': word_map}
    filename = 'checkpoint_grnn.pth.tar'
    torch.save(state, filename)


def adjust_learning_rate(optimizer, scale_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
