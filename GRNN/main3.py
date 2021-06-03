from preprocessing import GRNN_prep
from prep3 import GRNNDataset
from train3 import train
from test import test


data = GRNNDataset(csv_folder='./datasets/yahoo', output_folder='./grnn_data')
# Model architecture
sentence_model = 0  # {0: "convolution", 1: "lstm"}
gnn_output = 0  # {0: "last", 1: "avg"}
gnn_type = 0  # {0: "forward", 1: "forward-backward"} # if gnn_type=1 then gnn_output=1

# TRAIN
train(data, sentence_model, gnn_type, gnn_output)
# TEST
test(data, './grnn_data')
