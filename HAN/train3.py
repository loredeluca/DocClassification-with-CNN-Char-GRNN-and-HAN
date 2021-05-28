import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import datetime

from tqdm import tqdm
from preprocessing3 import HANDataset
from model3 import HierarchialAttentionNetwork
from utils3 import load_word2vec_embeddings, AverageMeter, save_checkpoint#, adjust_learning_rate


def HAN_train(data_folder, word_map, n_classes):
    """
    :param data_folder: folder where data files are stored
    :param word_map:
    :param n_classes:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    training_params = {"batch_size": 64,
                       "shuffle": True,
                       "num_workers": 4,
                       "pin_memory": True}

    # DataLoaders
    #training_set = HANDataset(data_folder, 'train')
    train_loader = DataLoader(HANDataset(data_folder, 'train'), **training_params)
    # VALIDATION
    validation_loader = DataLoader(HANDataset(data_folder, 'val'), batch_size=64)

    # Model parameters
    word2vec_file = os.path.join(data_folder, 'word2vec_model')  # path to pre-trained word2vec embeddings
    embeddings, emb_size = load_word2vec_embeddings(word2vec_file, word_map)  # load pre-trained word2vec embeddings
    word_rnn_size = 50  # word RNN size
    sentence_rnn_size = 50  # character RNN size
    word_rnn_layers = 1  # number of layers in character RNN
    sentence_rnn_layers = 1  # number of layers in word RNN
    word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
    sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
    dropout = 0.3  # dropout (serve per ridurre l'overfitting)
    fine_tune_word_embeddings = True  # fine-tune word embeddings?

    #model
    model = HierarchialAttentionNetwork(n_classes=n_classes, vocab_size=len(word_map), emb_size=emb_size,
                                        word_rnn_size=word_rnn_size, sentence_rnn_size=sentence_rnn_size,
                                        word_rnn_layers=word_rnn_layers, sentence_rnn_layers=sentence_rnn_layers,
                                        word_att_size=word_att_size, sentence_att_size=sentence_att_size, dropout=dropout)
    model.sentence_attention.word_attention.init_embeddings(embeddings)  # initialize embedding layer with pre-trained embeddings
    model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)  # fine-tune

    # Training parameters
    lr = 1e-3
    momentum = 0.9
    num_epochs = 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    total_t0 = time.time()
    loss_ = []
    for epoch in range(0, num_epochs):
        print('\n============================== Epoch {:} / {:} =============================='.format(epoch + 1,
                                                                                                       num_epochs))
        #batch_time = AverageMeter()  # forward prop. + back prop. time per batch
        #data_time = AverageMeter()  # data loading time per batch
        losses = AverageMeter()  # cross entropy loss
        #accs = AverageMeter()  # accuracies

        start = time.time()
        model.train()
        for step, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):

            #data_time.update(time.time() - start)

            documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
            sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
            words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(device)  # (batch_size)

            # Forward
            output, word_alphas, sentence_alphas = model(documents, sentences_per_document,words_per_sentence)
            # Loss
            loss = criterion(output, labels)
            # Back prop.
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            loss_.append(loss)

            # Find accuracy
            #_, predictions = output.max(dim=1)  # (n_documents)
            #correct_predictions = torch.eq(predictions, labels).sum().item()
            #accuracy = correct_predictions / labels.size(0)

            # Keep track of metrics
            losses.update(loss.item(), labels.size(0))
            #batch_time.update(time.time() - start)
            #accs.update(accuracy, labels.size(0))

            if step % 1000 == 0:
                print("Batch: [{:>5}/{:>5}]\t"
                      "Batch Time {}\t"
                      "Loss: {}".format(step + 1, len(train_loader), str(datetime.timedelta(
                                                                seconds=int(round(time.time() - start)))), loss))
                print("'LossMetodo: {loss.val:.4f} ({loss.avg:.4f})\t".format(loss=losses))

        # Decay learning rate every epoch
        # adjust_learning_rate(optimizer, 0.1)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, word_map)

        validation = True
        if validation is True:
            model.eval()
            accs = AverageMeter()

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

                    # Keep track of metrics
                    accs.update(accuracy, labels.size(0))
            print("Validation Accuracy: {}".format(accs.avg * 100))

    print("Loss:\n {}".format(loss_))
    print("END TRAINING\n")
    print("Total training time {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time() - total_t0))))))
