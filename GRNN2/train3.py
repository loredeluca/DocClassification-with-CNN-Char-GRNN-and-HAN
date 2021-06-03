import numpy as np
import os
import torch
import random
from torch.utils.data import sampler

#import gensim as gs
from prep3 import CustomDataloader
from model3 import DocSenModel


def split_data(dataset, validation_split, shuffle_dataset=True, random_seed=3):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def train(dataset, sentence_model, gnn_type, gnn_output, validation_split, model_name,
          model_path, continue_training=True, early_stopping=2):
    # parameters
    num_epochs = 70
    learning_rate = 0.03
    lr_decay_factor = 0.8
    l2_reg = 1e-5
    batch_size = 50

    checkpoint_path = model_path + '_checkpoint.tar'
    if continue_training and os.path.isfile(checkpoint_path):
        print("Loading checkpoint to continue training...")
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_0 = checkpoint['epoch'] + 1
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        train_acc = checkpoint['train_acc']
        valid_acc = checkpoint['valid_acc']
        train_indices = checkpoint['train_indices']
        val_indices = checkpoint['val_indices']
        try:
            if learning_rate == 0 or lr_decay_factor == 0:
                learning_rate = checkpoint['learning_rate']
                lr_decay_factor = checkpoint['lr_decay_factor']
                learning_rate = learning_rate * lr_decay_factor
        except Exception:
            pass
        print(f"Continue training in epoch {epoch_0+1} with learning rate {learning_rate}")
    else:
        print("Not loading a training checkpoint.")
        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []
        epoch_0 = 0
        train_indices, val_indices = split_data(dataset, validation_split)

    if sentence_model == 0:
        print("Sentence model: convolution")
        model_name += '-conv'
        sentence_model = DocSenModel.SentenceModel.CONV
    else:
        print("Sentence model: LSTM")
        model_name += '-lstm'
        sentence_model = DocSenModel.SentenceModel.LSTM

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

    freeze_embedding = True
    model = DocSenModel(dataset.num_classes, sentence_model, gnn_output, gnn_type, dataset.embedding, freeze_embedding,
                        cuda=True)

    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    dataloader_train = CustomDataloader(batch_size, train_sampler, dataset)
    dataloader_valid = CustomDataloader(batch_size, valid_sampler, dataset)

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
        if os.path.isfile(checkpoint_path):
            os.rename(checkpoint_path, checkpoint_path + '_' + str(epoch))
        torch.save({
            'epoch': epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'learning_rate': learning_rate,
            'lr_decay_factor': lr_decay_factor
        }, checkpoint_path + '_tmp')
        os.rename(checkpoint_path + '_tmp', checkpoint_path)

        # Update Learning Rate
        learning_rate = lr_decay_factor * learning_rate
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
