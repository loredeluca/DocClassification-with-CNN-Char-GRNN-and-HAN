import numpy as np
import os
import torch
import torch.nn as nn
import random
from torch.utils.data import sampler

from prep5 import GRNNDataset2, CustomDataloader2
from model5 import DocSenModel


def train(data_folder, embedding, sentence_model, model_name, train_size, val_size, early_stopping=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 50
    training_set = GRNNDataset2(data_folder, 'train')
    dataloader_train = CustomDataloader2(batch_size, training_set, train_size)

    valid_set = GRNNDataset2(data_folder, 'val')
    val_indices = [*range(0, val_size, 1)]
    dataloader_valid = CustomDataloader2(batch_size, valid_set, val_size)

    if sentence_model == 0:
        print("Sentence model: convolution")
        model_name += '-conv'
        sentence_model = DocSenModel.SentenceModel.CONV
    else:
        print("Sentence model: LSTM")
        model_name += '-lstm'
        sentence_model = DocSenModel.SentenceModel.LSTM
    '''
    if gnn_type == 0:
        print("GNN type: forward")
        model_name += '-forward'
        gnn_type = DocSenModel.GnnType.FORWARD
    else:
        print("GNN type: forward-backward")
        model_name += '-forward-backward'
        gnn_type = DocSenModel.GnnType.FORWARD_BACKWARD

    if gnn_output == 0:
        print("GNN output: last")
        model_name += '-last'
        gnn_output = DocSenModel.GnnOutput.LAST
    else:
        print("GNN output: avg")
        model_name += '-avg'
        gnn_output = DocSenModel.GnnOutput.AVG
    '''

    model = DocSenModel(training_set.n_classes, sentence_model, embedding, device)

    # parameters
    num_epochs = 70
    learning_rate = 0.03
    lr_decay_factor = 0.8
    l2_reg = 1e-5

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    epoch_0 = 0
    # train_indices, val_indices = split_data(dataset, validation_split=0.2)

    min_loss = 99999
    min_loss_epoch = epoch_0
    for epoch in range(epoch_0, num_epochs):
        if early_stopping > 0:
            if epoch - min_loss_epoch >= early_stopping:
                print(f"No training improvement over the last {early_stopping} epochs. Aborting.")
                break

        for batch_num, batch in enumerate(dataloader_train):
            # Forward pass for each single document in the batch
            predictions = None
            labels = None
            matches = 0
            for (doc, label) in batch:
                if len(doc) == 0:
                    continue

                prediction = model(doc)
                prediction = prediction.unsqueeze(0)
                predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                label = torch.Tensor([label])
                label = label.long().to(model._device)
                labels = label if labels is None else torch.cat((labels, label))

                if label == torch.argmax(prediction):
                    matches += 1

            # Compute the loss
            loss_object = loss_function(predictions, labels)
            loss = loss_object.item()
            train_loss.append(loss)

            accuracy = float(matches) / float(len(predictions))
            train_acc.append(accuracy)

            if loss < min_loss:
                min_loss = loss
                min_loss_epoch = epoch

            # Reset the gradients in the optimizer.
            # Otherwise past computations would influence new computations.
            optimizer.zero_grad()
            loss_object.backward()
            optimizer.step()

            # Test on a single batch from the validation set
            # Set model to evaluation mode
            model.eval()
            batch_valid_indices = random.choices(val_indices, k=batch_size)
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
            loss_object = loss_function(predictions, labels)
            valid_loss.append(loss_object.item())

            accuracy = float(matches) / float(len(predictions))
            valid_acc.append(accuracy)

            # Set model back to training mode
            model.train()

            if batch_num % 10 == 0:
                print(f"  Epoch {epoch+1:>2} of {num_epochs} - Batch {batch_num+1:>5} of {len(dataloader_train):>5}  -"
                      f" lr {learning_rate} -"
                      f"   Tr.-Loss: {train_loss[-1]:.4f}   Val.-Loss: {valid_loss[-1]:.4f}"
                      f"   Tr.-Acc.: {train_acc[-1]:.2f}   Val.-Acc.: {valid_acc[-1]:.2f}")

        print("Saving training progress checkpoint...")
        torch.save({
            'epoch': epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'val_indices': val_indices,
            'learning_rate': learning_rate,
            'lr_decay_factor': lr_decay_factor
        }, 'checkpoint_GRNN_prova.pth.tar')

        # Update Learning Rate
        learning_rate = lr_decay_factor * learning_rate
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
