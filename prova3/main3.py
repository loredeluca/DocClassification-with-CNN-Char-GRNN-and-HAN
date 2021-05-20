from preprocessing3 import preprocessing, train_word2vec_model
from train3 import train
from test3 import test

data_name = "yahoo"
classes = preprocessing(csv_folder='./datasets/yahoo', output_folder='./han_data', sentence_limit=15, word_limit=20, min_word_count=5)
train_word2vec_model(data_folder='./han_data')#, algorithm='skipgram')
# algorithm si può levare
print("termina preprocessing")
train(classes)
print("Start Test")
test()


