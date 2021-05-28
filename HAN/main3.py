from preprocessing3 import HAN_preprocessing
from train3 import HAN_train
from test3 import HAN_test

data_name = "yahoo"
# si usa sentence limit e word limit, per avere tutte frasi della stessa lunghezza
word_vocab, classes = HAN_preprocessing(csv_folder='./datasets/yahoo', output_folder='./han_data', sentence_limit=15, word_limit=20, min_word_count=5)
print("START TRAINING\n")
HAN_train(data_folder='./han_data', word_map=word_vocab, n_classes=classes)
print("START TESTING\n")
HAN_test(data_folder='./han_data')

