from prep5 import GRNN_preprocess2
from train5 import train
#from test4 import test


data_name = 'yelp'
model_name = 'gnn-yelp'

sentence_model, gnn_type, gnn_output = 1, 1, 1

# data_path = './data/yelp_full.csv'
model_path = './models/' + model_name
w2v_path = './data/Word2Vec/'
prep_path = './data/Preprocessed/'


data_folder = './datasets/yelp'
output_folder = './grnn_data'

embedding, word_map, train_size, val_size, test_size = GRNN_preprocess2(data_folder, output_folder)
print("START TRAINING\n")
train(output_folder, embedding, sentence_model, model_name, train_size, val_size)
print("START TEST\n")



#dataset = preprocess(csv_folder='./data/yelp_full.csv', data_name='yelp', w2v_path=w2v_path, prep_path=prep_path)
#train(dataset, sentence_model, gnn_type, gnn_output, model_name, model_path)
#test(dataset, model_path)

