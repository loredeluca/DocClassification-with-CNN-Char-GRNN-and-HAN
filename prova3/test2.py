import torch
import numpy as np

from train2 import gen_batch

def test_accuracy(batch_size, x_test, y_test, sent_attn_model):
    acc = []
    test_length = len(x_test)
    for j in range(int(test_length / batch_size)):
        x, y = gen_batch(x_test, y_test, batch_size)
        state_word = sent_attn_model.init_hidden_word()
        state_sent = sent_attn_model.init_hidden_sent()

        y_pred, state_sent = sent_attn_model(x, state_sent, state_word)
        max_index = y_pred.max(dim=1)[1]
        correct = (max_index == torch.cuda.LongTensor(y)).sum()
        acc.append(float(correct) / batch_size)
    return np.mean(acc)