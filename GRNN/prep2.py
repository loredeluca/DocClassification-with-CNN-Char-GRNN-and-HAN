import pandas as pd
import os
import torch
from tqdm import tqdm
import re
import string
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import logging
from torch.utils.data import Dataset


class GRNNDataset(Dataset):
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        split = split.upper()
        assert split in {'TRAIN', 'VAL', 'TEST'}
        self.split = split

        # Load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))
        self.classes = sorted([int(y) for y in set(self.data['labels'])])

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, i):
        return self.data['docs'][i], self.classes.index(int(self.data['labels'][i]))
        # return torch.LongTensor(self.data['docs'][i]), torch.LongTensor([self.data['sentences_per_document'][i]]), \
        #       torch.LongTensor(self.data['words_per_sentence'][i]), torch.LongTensor([self.data['labels'][i]])



def GRNN_prep2(csv_folder, output_folder, sentence_limit, word_limit, save_word2vec_data=True):
    print('Reading and tokenize training data...\n')
    # TRAIN
    train_texts, train_labels, n_classes = read_csv2(csv_folder, 'train', sentence_limit, word_limit)

    # Save text data for word2vec
    if save_word2vec_data:
        torch.save(train_texts, os.path.join(output_folder, 'word2vec_data.pth.tar'))
        print('\nText data for word2vec saved to %s.\n' % os.path.abspath(output_folder))

    print('\n Train word2vec model...')
    train_word2vec_model2(data_folder=output_folder)
    print('\nEND TRAINING WORD2VEC MODEL\n')

    print("Loading word2vec model...")
    embedding, word2index = load_word2vec_embeddings2(output_folder)
    index2word = {index: word for (word, index) in word2index.items()}

    train_texts_index = words_to_vocab_index2(train_texts, word2index)

    # Save
    print('Saving data train...\n')
    torch.save({'docs': train_texts_index, 'labels': train_labels}, os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Training data saved to %s.\n' % os.path.abspath(output_folder))

    # VAL
    eval_texts, eval_labels, _ = read_csv2(csv_folder, 'val', sentence_limit, word_limit)
    eval_texts_index = words_to_vocab_index2(eval_texts, word2index)
    # Save
    print('Saving...\n')
    torch.save({'docs': eval_texts_index, 'labels': eval_labels}, os.path.join(output_folder, 'VAL_data.pth.tar'))
    print('Val data saved to %s.\n' % os.path.abspath(output_folder))

    # TEST
    test_texts, test_labels, _ = read_csv2(csv_folder, 'test', sentence_limit, word_limit)
    test_texts_index = words_to_vocab_index2(test_texts, word2index)
    # Save
    print('Saving...\n')
    torch.save({'docs': test_texts_index, 'labels': test_labels}, os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Val data saved to %s.\n' % os.path.abspath(output_folder))

    print("END PREPROCESSING!\n")

    return embedding, word2index, index2word, n_classes


def read_csv2(data_folder, split, sentence_limit, word_limit):
    split = split.lower()
    assert split in {'train', 'val', 'test'}

    data = pd.read_csv(os.path.join(data_folder, split + '.csv'))  # , header=None)

    x_data = data['text']
    y_data = data['label']
    x_data_prep = []

    for i, doc in enumerate(tqdm(x_data)):
        x_data_prep.append([])
        split = re.split('\!|\.|\?|\;|\:|\n', doc)
        for sentence in split:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            x_data_prep[-1].append(simple_preprocess(sentence, min_len=1, max_len=20, deacc=True))

    n_classes = len(np.unique(y_data))

    return x_data_prep, y_data, n_classes


def train_word2vec_model2(data_folder):
    # Read data
    docs = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))
    sentences = [sentence for doc in docs for sentence in doc]

    # print intermediate info
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize and train the model (this will take some time)
    word2vec = Word2Vec(sentences=sentences, vector_size=200, window=5)

    word2vec.wv.save(os.path.join(data_folder, 'word2vec_model'))


def load_word2vec_embeddings2(data_folder):
    # Load word2vec model into memory
    word2vec_file = os.path.join(data_folder, 'word2vec_model')
    w2v = KeyedVectors.load(word2vec_file, mmap='r')

    # embedding matrix is orderd by indices in model.wv.voacab
    word2index = {token: token_index for token_index, token in enumerate(w2v.index_to_key)}

    # embedding = np.load(w2v_word_vectors_path)
    embedding = w2v.wv.vectors
    unknown_vector = np.mean(embedding, axis=0)
    padding_vector = np.zeros(len(embedding[0]))

    embedding = np.append(embedding, unknown_vector.reshape((1, -1)), axis=0)
    embedding = np.append(embedding, padding_vector.reshape((1, -1)), axis=0)

    word2index['<unk>'] = len(embedding) - 2  # map unknown words to vector we just appended
    word2index['<pad>'] = len(embedding) - 1

    return embedding, word2index

def words_to_vocab_index2(documents, word2index):
    """
    Replace each word in the training data by it´s index in the vocab
    :param documents: List of documents containing lists of sentences containing lists of words
    :param word2index: Dict word -> vocab index
    :return: List of documents containing lists of sentences containing lists of vocabulary ids
    """
    documents_ind = []
    for document in documents:
        document_ind = []
        documents_ind.append(document_ind)
        for sentence in document:
            sentence_ind = []
            document_ind.append(sentence_ind)
            for word in sentence:
                if word in word2index:
                    sentence_ind.append(word2index[word])
                else:
                    sentence_ind.append(word2index['<unk>'])
    return documents_ind
