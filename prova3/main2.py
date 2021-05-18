import torch

from preprocessing2 import hanPreprocessing
from train2 import train
from test2 import test_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print(" ")
dataname = "twitter"
print("start preprocessing")
print(" ")
X_train_pad, X_val_pad, X_test_pad, y_train_tensor, y_val_tensor, y_test_tensor, vocab_size, classes, weights, max_seq_len = hanPreprocessing(dataname)
print("end preprocessing")

batch_size = 64
print("start Train+Val")
sent_attn = train(X_train_pad, X_val_pad, y_train_tensor, y_val_tensor, batch_size, vocab_size, classes, weights, max_seq_len)
print("end Train+Val")
print("start Test")
test_accuracy(batch_size, X_test_pad, y_test_tensor, sent_attn)
print("end Test")
