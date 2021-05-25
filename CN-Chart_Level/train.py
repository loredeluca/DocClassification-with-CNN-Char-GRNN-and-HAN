import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time

from model import CharCNN
from utils import shuffle_matrix, mini_batch_generator


def train(train_docs, train_labels, test_docs, test_labels, vocab, alphabet_size, check, maxlen):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # Training parameters
    lr = 0.01  # learning rate
    momentum = 0.9  # momentum
    epochs = 10  # number of epochs to run
    batch_size = 50

    model = CharCNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    t0 = time.time()
    for epoch in range(epochs):
        doc_train, labels_train = shuffle_matrix(train_docs, train_labels)

        batches = mini_batch_generator(doc_train, labels_train, vocab, alphabet_size,
                                                        check, maxlen, batch_size=batch_size)

        doc_test, labels_test = shuffle_matrix(test_docs, test_labels)

        test_batches = mini_batch_generator(doc_test, labels_test, vocab, alphabet_size,
                                            check, maxlen, batch_size=1)

        accuracy = 0.0

        step = 1
        start = time.time()
        print('Epoch: {}'.format(epoch))
        running_loss = 0.0
        i = 0
        train_loss_avg = 0.0
        train_correct = 0.0
        for step, (x_train, y_train) in enumerate(batches):
            i = i + 1
            inputs = torch.from_numpy(np.swapaxes(x_train.astype(np.float64), 1, 2))
            labels = torch.from_numpy(np.argmax(y_train, axis=1).astype(np.float64))
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            inputs = inputs.float()
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # train_loss += loss.data[0]
            train_loss_avg = loss.data.mean()
            _, predicted = torch.max(outputs.data, 1)
            train_correct = (predicted == labels.data).sum()
            accuracy += (train_correct * 100 / len(labels))
            accuracy_avg = accuracy / step
            if step % 100 == 0:
                print('  Step: {}, Batch Time {}'.format(step, time.time()-start))
                print('\tLoss: {}. Accuracy: {}'.format(train_loss_avg, accuracy_avg))
            #step += 1

        # ---
        test_accuracy = 0
        test_loss_avg = 0
        test_correct = 0
        test_accuracy_avg = 0
        for step, (x_test_batch, y_test_batch) in enumerate(test_batches):
            inputs = torch.from_numpy(np.swapaxes(x_test_batch.astype(np.float64), 1, 2))
            labels = torch.from_numpy(np.argmax(y_test_batch, axis=1).astype(np.float64))
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            inputs = inputs.float()
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss_avg += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            test_correct = (predicted == labels.data).sum()
            test_accuracy += (test_correct * 100 / len(labels))
            # test_accuracy_avg += (test_accuracy /step)
            # step += 1
        print('Test Loss: {}. Test Accuracy: {}. Test Time: {}'.format(test_loss_avg / step, test_accuracy / step,
                                                                       time.time() - start))


    print("END TRAINING, Time: {}".format(t0-time.time()))



