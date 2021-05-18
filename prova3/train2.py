import torch
from torch import nn
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from model2 import SentenceRNN

def train_data(batch_size, review, targets, sent_attn_model, sent_optimizer, criterion):
    state_word = sent_attn_model.init_hidden_word()
    state_sent = sent_attn_model.init_hidden_sent()
    sent_optimizer.zero_grad()

    y_pred, state_sent = sent_attn_model(review, state_sent, state_word)

    loss = criterion(y_pred.cuda(), torch.cuda.LongTensor(targets))

    max_index = y_pred.max(dim=1)[1]
    correct = (max_index == torch.cuda.LongTensor(targets)).sum()
    acc = float(correct) / batch_size

    loss.backward()

    sent_optimizer.step()

    return loss.data[0], acc


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def gen_batch(x,y,batch_size):
    k = random.sample(range(len(x)-1),batch_size)
    x_batch=[]
    y_batch=[]

    for t in k:
        x_batch.append(x[t])
        y_batch.append(y[t])

    return [x_batch,y_batch]


def validation_accuracy(batch_size, x_val, y_val, sent_attn_model):
    acc = []
    val_length = len(x_val)
    for j in range(int(val_length / batch_size)):
        x, y = gen_batch(x_val, y_val, batch_size)
        state_word = sent_attn_model.init_hidden_word()
        state_sent = sent_attn_model.init_hidden_sent()

        y_pred, state_sent = sent_attn_model(x, state_sent, state_word)
        max_index = y_pred.max(dim=1)[1]
        correct = (max_index == torch.cuda.LongTensor(y)).sum()
        acc.append(float(correct) / batch_size)
    return np.mean(acc)


def train_early_stopping(batch_size, x_train, y_train, x_val, y_val, sent_attn_model,
                         sent_attn_optimiser, loss_criterion, num_epoch,
                         print_loss_every=50, code_test=True):
    start = time.time()
    loss_full = []
    loss_epoch = []
    acc_epoch = []
    acc_full = []
    val_acc = []
    epoch_counter = 0
    train_length = len(x_train)
    for epoch in range(1, num_epoch + 1):
        loss_epoch = []
        acc_epoch = []
        for j in range(int(train_length / batch_size)):
            x, y = gen_batch(x_train, y_train, batch_size)
            loss, acc = train_data(batch_size, x, y, sent_attn_model, sent_attn_optimiser, loss_criterion)
            loss_epoch.append(loss)
            acc_epoch.append(acc)
            if (code_test and j % int(print_loss_every / batch_size) == 0):
                print('Loss at %d paragraphs, %d epoch,(%s) is %f' % (
                j * batch_size, epoch, timeSince(start), np.mean(loss_epoch)))
                print('Accuracy at %d paragraphs, %d epoch,(%s) is %f' % (
                j * batch_size, epoch, timeSince(start), np.mean(acc_epoch)))

        loss_full.append(np.mean(loss_epoch))
        acc_full.append(np.mean(acc_epoch))
        torch.save(sent_attn_model.state_dict(), 'sent_attn_model_yelp.pth')
        print('Loss after %d epoch,(%s) is %f' % (epoch, timeSince(start), np.mean(loss_epoch)))
        print('Train Accuracy after %d epoch,(%s) is %f' % (epoch, timeSince(start), np.mean(acc_epoch)))

        val_acc.append(validation_accuracy(batch_size, x_val, y_val, sent_attn_model))
        print('Validation Accuracy after %d epoch,(%s) is %f' % (epoch, timeSince(start), val_acc[-1]))
    return loss_full, acc_full, val_acc

def plotLoss(loss_full):
    plt.plot(loss_full)
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')
    plt.savefig('loss.png')


def plotFullAcc(acc_full):
    plt.plot(acc_full)
    plt.ylabel('Training Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('train_acc.png')


def plotValAcc(val_acc):
    plt.plot(val_acc)
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('val_acc.png')


def train(X_train_pad, X_val_pad, y_train_tensor, y_val_tensor, batch_size, vocab_size, classes, weights, max_seq_len):

    hid_size = 100
    embedsize = 200

    # model
    sent_attn = SentenceRNN(vocab_size, embedsize, batch_size, hid_size, classes, max_seq_len)
    sent_attn.cuda()
    sent_attn.wordRNN.embed.from_pretrained(weights)
    #torch.backends.cudnn.benchmark = True

    learning_rate = 1e-3
    momentum = 0.9

    # loss function
    criterion = nn.NLLLoss()
    sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=learning_rate, momentum=momentum)

    epoch = 200

    loss_full, acc_full, val_acc = train_early_stopping(batch_size, X_train_pad, y_train_tensor, X_val_pad,
                                                        y_val_tensor, sent_attn, sent_optimizer, criterion, epoch,
                                                        10000, False)
    plotLoss(loss_full)
    plotFullAcc(acc_full)
    plotValAcc(val_acc)

    return sent_attn
