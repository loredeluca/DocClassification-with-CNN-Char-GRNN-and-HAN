from prep4 import GRNN_preprocess
from train4 import train
#from test4 import test


data_name = 'yelp'
model_name = 'gnn-yelp'

sentence_model, gnn_type, gnn_output = 0, 0, 0

# data_path = './data/yelp_full.csv'
model_path = './models/' + model_name
w2v_path = './data/Word2Vec/'
prep_path = './data/Preprocessed/'


data_folder = './datasets/yahoo'
output_folder = './han_data'

word_vocab, classes = GRNN_preprocess(data_folder, output_folder, sentence_limit=15, word_limit=20)
print("START TRAINING\n")
train(output_folder, classes, word_vocab, sentence_model, gnn_type, gnn_output, model_name)


#dataset = preprocess(csv_folder='./data/yelp_full.csv', data_name='yelp', w2v_path=w2v_path, prep_path=prep_path)
#train(dataset, sentence_model, gnn_type, gnn_output, model_name, model_path)
#test(dataset, model_path)

