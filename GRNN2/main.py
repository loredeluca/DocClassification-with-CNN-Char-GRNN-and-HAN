from prep3 import YelpDataset
from train3 import train
from test3 import test


def main():
    validation_split = 0.2

    data_name = 'yelp'
    model_name = 'gnn-yelp'

    sentence_model, gnn_type, gnn_output = 0, 0, 0

    data_path = './data/yelp2013/yelp_full.csv'
    model_path = './models/' + model_name
    w2v_path = './data/Word2Vec/'
    prep_path = './data/Preprocessed/'

    dataset = YelpDataset(data_path, data_name, w2v_path=w2v_path, prep_path=prep_path, w2v_sample_frac=0.9)

    train(dataset, sentence_model, gnn_type, gnn_output, validation_split, model_name, model_path)

    test(dataset, model_path)


if __name__ == '__main__':
    main()
