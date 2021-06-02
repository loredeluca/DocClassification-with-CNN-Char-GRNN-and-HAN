from preprocessing import GRNN_prep
from train import train
from test import test


train_texts, train_labels, embedding, word2index, n_classes = GRNN_prep(csv_folder='./datasets/yahoo', output_folder='./grnn_data', sentence_limit=15, word_limit=20)

# Model architecture
sentence_model = 0  # {0: "convolution", 1: "lstm"}
gnn_output = 0  # {0: "last", 1: "avg"}
gnn_type = 0  # {0: "forward", 1: "forward-backward"} # if gnn_type=1 then gnn_output=1

# TRAIN
train('./grnn_data', n_classes, embedding, sentence_model, gnn_type, gnn_output)
# TEST
test('./grnn_data')
