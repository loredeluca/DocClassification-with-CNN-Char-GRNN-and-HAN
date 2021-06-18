import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import datetime
import os
import random
from statistics import mean


from CNNChar_preprocessing import CN_CharDataset
from CNNChar_model import CharacterLevelCNN
from HAN_preprocessing import HANDataset
from HAN_model import HierarchialAttentionNetwork
from GRNN_preprocessing import GRNNDataset, CustomDataloader
from GRNN_model import GRNN
from utils import load_word2vec_embeddings_han, adjust_learning_rate, save_checkpoint2



def CNNChart_train(data_folder, feature):
    """
    :param data_folder: folder where data files are stored
    :param feature: ConvNets choice 'large' or 'small'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    training_params = {"batch_size": 64,
                       "shuffle": True,
                       "num_workers": 0}
    # DataLoader
    training_set = CN_CharDataset(data_folder, 'train')
    train_loader = DataLoader(training_set, **training_params)

    # Model
    if feature == "small":
        model = CharacterLevelCNN(input_length=training_set.max_length, n_classes=training_set.n_classes,
                                  input_dim=len(training_set.alphabet), n_conv_filters=256, n_fc_neurons=1024)

    elif feature == "large":
        model = CharacterLevelCNN(input_length=training_set.max_length, n_classes=training_set.n_classes,
                                  input_dim=len(training_set.alphabet), n_conv_filters=1024, n_fc_neurons=2048)
    # Training parameters
    lr = 1e-3
    momentum = 0.9
    num_epochs = 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    t0 = time.time()
    loss_ = []
    for epoch in range(0, num_epochs):
        print('\n============================== Epoch {:} / {:} =============================='.format(epoch + 1,
                                                                                                       num_epochs))
        start = time.time()
        model.train()
        for step, (data, label) in enumerate(train_loader):
            feature = data.to(device)
            label = label.to(device)

            # Forward
            output = model(feature)
            # Loss
            loss = criterion(output, label)
            # Back prop.
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            loss_.append(loss)

            if step % 100 == 0:
                print("Batch: [{:>5}/{:>5}]\t"
                      "Batch Time {}\t"
                      "Loss: {}".format(step + 1, len(train_loader), str(datetime.timedelta(
                                                                  seconds=int(round(time.time() - start)))), loss))
        # Save checkpoint
        save_checkpoint2('cnn-chart', epoch, model, optimizer)
    # print("Loss:\n {}".format(loss_))
    print("END TRAINING")
    print("Total training time: {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time() - t0))))))


def HAN_train(data_folder, word_map, n_classes):
    """
    :param data_folder: folder where data files are stored
    :param word_map: word map from Word2Vec
    :param n_classes: number of classes in the dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    training_params = {"batch_size": 64,
                       "shuffle": True,
                       "num_workers": 4,
                       "pin_memory": True}

    # DataLoaders
    training_set = HANDataset(data_folder, 'train')
    train_loader = DataLoader(training_set, **training_params)

    val_set = HANDataset(data_folder, 'val')
    validation_loader = DataLoader(val_set, batch_size=64)

    # Model parameters
    word2vec_file = os.path.join(data_folder, 'word2vec_model')
    embeddings, emb_size = load_word2vec_embeddings_han(word2vec_file, word_map)

    # Model
    model = HierarchialAttentionNetwork(n_classes=n_classes, vocab_size=len(word_map), emb_size=emb_size,
                                        word_rnn_size=50, sentence_rnn_size=50,
                                        word_rnn_layers=1, sentence_rnn_layers=1,
                                        word_att_size=100, sentence_att_size=100, dropout=0.3)
    # initialize embedding layer with pre-trained embeddings
    model.sentence_attention.word_attention.init_embeddings(embeddings)
    model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune=True)

    # Training parameters
    lr = 1e-3
    num_epochs = 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    t0 = time.time()
    loss_, acc_ = [], []
    for epoch in range(0, num_epochs):
        print('\n============================== Epoch {:} / {:} =============================='.format(epoch + 1,
                                                                                                       num_epochs))
        start = time.time()
        model.train()
        for step, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):
            documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
            sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
            words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(device)  # (batch_size)

            # Forward
            output, word_alphas, sentence_alphas = model(documents, sentences_per_document, words_per_sentence)
            # Loss
            loss = criterion(output, labels)
            # Back prop.
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            loss_.append(loss)

            if step % 1000 == 0:
                print("Batch: [{:>5}/{:>5}]\t"
                      "Batch Time {}\t"
                      "Loss: {}".format(step + 1, len(train_loader), str(datetime.timedelta(
                                                                seconds=int(round(time.time() - start)))), loss))

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, 0.1)

        # Save checkpoint
        save_checkpoint2('han', epoch, model, optimizer, word_map)

        # Validation
        if True:
            model.eval()
            with torch.no_grad():
                for documents, sentences_per_document, words_per_sentence, labels in validation_loader:
                    documents = documents.to(device)
                    sentences_per_document = sentences_per_document.squeeze(1).to(device)
                    words_per_sentence = words_per_sentence.to(device)
                    labels = labels.squeeze(1).to(device)

                    # Forward
                    out, _, _ = model(documents, sentences_per_document, words_per_sentence)

                    # Find accuracy
                    _, predictions = out.max(dim=1)
                    correct_predictions = torch.eq(predictions, labels).sum().item()
                    accuracy = correct_predictions / labels.size(0)

                    acc_.append(accuracy)

            print("Validation Accuracy: {0:.2f}%".format(mean(acc_) * 100))

    # print("Loss:\n {}".format(loss_))
    print("END TRAINING\n")
    print("Total training time: {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time() - t0))))))


def GRNN_train(data_folder, embedding, sentence_model, train_size, val_size, early_stopping=2):
    """
    :param data_folder: folder where data files are stored
    :param embedding: embedding from Word2Vec
    :param sentence_model: convolution or LSTM
    :param train_size: test set size
    :param val_size: val set size
    :param early_stopping: number of epochs without improvement
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # DataLoader
    training_set = GRNNDataset(data_folder, 'train')
    train_loader = CustomDataloader(training_set, train_size)

    val_set = GRNNDataset(data_folder, 'val')
    val_indices = [*range(0, val_size, 1)]
    val_loader = CustomDataloader(val_set, val_size)

    # Model parameters
    if sentence_model == 0:
        print("Sentence model: convolution")
        sentence_model = GRNN.SentenceModel.CONV
    else:
        print("Sentence model: LSTM")
        sentence_model = GRNN.SentenceModel.LSTM

    model = GRNN(training_set.n_classes, sentence_model, embedding, device)

    # Training parameters
    lr = 0.03
    num_epochs = 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)

    min_loss = 99999
    min_loss_epoch = 0

    t0 = time.time()
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    for epoch in range(0, num_epochs):
        if early_stopping > 0:
            if epoch - min_loss_epoch >= early_stopping:
                print(f"No training improvement over the last {early_stopping} epochs. Aborting.")
                break
        print('\n============================== Epoch {:} / {:} =============================='.format(epoch + 1,
                                                                                                       num_epochs))
        start = time.time()
        for step, batch in enumerate(train_loader):
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
                label = label.long().to(device)
                labels = label if labels is None else torch.cat((labels, label))

                if label == torch.argmax(prediction):
                    matches += 1

            # Loss
            loss = criterion(predictions, labels)

            # Back prop.
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            train_loss.append(loss.item())

            if step % 10 == 0:
                print("Batch: [{:>5}/{:>5}]\t"
                      "Batch Time {}\t"
                      "lr: {} - Loss: {}".format(step + 1, len(train_loader), str(datetime.timedelta(
                                                                seconds=int(round(time.time() - start)))), lr, train_loss[-1]))

            if loss.item() < min_loss:
                min_loss = loss.item()
                min_loss_epoch = epoch

            # Decay learning rate every epoch
            adjust_learning_rate(optimizer, 0.8)

            # Save checkpoint
            save_checkpoint2('grnn', epoch, model, optimizer)

            # Validation
            model.eval()
            batch_valid_indices = random.choices(val_indices, k=64)
            predictions = None
            labels = None
            matches = 0
            for (doc, label) in val_loader.batch_iterator(batch_valid_indices):
                if len(doc) == 0:
                    continue
                try:
                    prediction = model(doc)
                    prediction = prediction.unsqueeze(0)
                    predictions = prediction if predictions is None else torch.cat((predictions, prediction))
                    label = torch.Tensor([label])
                    label = label.long().to(device)
                    labels = label if labels is None else torch.cat((labels, label))

                    if label == torch.argmax(prediction):
                        matches += 1
                except (AttributeError, RuntimeError) as e:
                    print("Some error occurred. Ignoring this document. Error:")
                    print(e)
                    continue

            accuracy = float(matches) / float(len(predictions))
            valid_acc.append(accuracy)

            # Set model back to training mode
            model.train()

            if step % 10 == 0:
                print("Validation Accuracy: {0:.2f}%".format(valid_acc[-1]*100))

    # print("Loss:\n {}".format(train_loss))
    print("END TRAINING\n")
    print("Total training time: {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time() - t0))))))

