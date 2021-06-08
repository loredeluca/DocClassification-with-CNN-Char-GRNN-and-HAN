import numpy as np
import pandas as pd
import string
import re

from typing import List
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess

import logging
from torch.utils.data import Dataset, sampler

import os
from tqdm import tqdm

import torch



def prep_data(data_folder, split):
    data = pd.read_csv(os.path.join(data_folder, split + '.csv'))
    data_size = data.shape[0]

    y_data = data['label']
    X_data_text = data['text']
    # Separate and preprocess words in sentences
    X_data_prep = []
    for i, doc in enumerate(tqdm(X_data_text)):
        X_data_prep.append([])
        split = re.split('\!|\.|\?|\;|\:|\n', doc)
        for sentence in split:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            X_data_prep[-1].append(simple_preprocess(sentence, min_len=1, max_len=20, deacc=True))

    return X_data_prep, y_data, data_size


def GRNN_preprocess2(csv_folder, output_folder, save_word2vec_data=True):
    # Read and Preprocessing train data
    print('Reading and preprocessing train_data...\n')
    train_texts, train_labels, train_size = prep_data(csv_folder, 'train')

    # Save text data for word2vec
    if save_word2vec_data:
        torch.save(train_texts, os.path.join(output_folder, 'word2vec_data.pth.tar'))

    # Read and Preprocessing val data
    print('Reading and preprocessing val_data...\n')
    val_texts, val_labels, val_size = prep_data(csv_folder, 'val')

    # Read and Preprocessing test data
    print('Reading and preprocessing test_data...\n')
    test_texts, test_labels, test_size = prep_data(csv_folder, 'test')

    print('\n Train word2vec model...')
    train_word2vec_model2(data_folder=output_folder)
    print('\nEND TRAINING WORD2VEC MODEL\n')

    # Build word_map & embedding
    embedding, word_map = load_word2vec_embeddings2(output_folder)

    # Encode train data
    encode_data('train', train_texts, word_map, train_labels, output_folder)
    # Encode val data
    encode_data('val', val_texts, word_map, val_labels, output_folder)
    # Encode test data
    encode_data('test', test_texts, word_map, test_labels, output_folder)

    print('END PREPROCESSING!\n')
    return embedding, word_map, train_size, val_size, test_size


def encode_data(split, documents, word_map, labels, output_folder):
    """
    Replace each word in the training data by it´s index in the vocab
    :param documents: List of documents containing lists of sentences containing lists of words
    :param word2index: Dict word -> vocab index
    :return: List of documents containing lists of sentences containing lists of vocabulary ids
    """
    documents_encoded = []
    for document in documents:
        document_encoded = []
        documents_encoded.append(document_encoded)
        for sentence in document:
            sentence_encoded = []
            document_encoded.append(sentence_encoded)
            for word in sentence:
                if word in word_map:
                    sentence_encoded.append(word_map[word])
                else:
                    sentence_encoded.append(word_map['<UNK>'])

    # Save
    print('Saving preprocessed ', split, '_texts...\n')
    torch.save({'docs': documents_encoded,
                'labels': labels},
               os.path.join(output_folder, split + '_data.pth.tar'))



def train_word2vec_model2(data_folder):
    # Sviluppiamo un word embeddings addestrando i nostri modelli word2vec su un corpus personalizzato
    # l'alternativa è usare un dataset pre-trained(Glove), in cui ad ogni parola è associata una stringa numerica
    # Train a word2vec model for word embeddings.
    # See the paper by Mikolov et. al. for details - https://arxiv.org/pdf/1310.4546.pdf

    # Read data
    docs = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))
    sentences = [sentence for doc in docs for sentence in doc]

    # print intermediate info
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    word2vec = Word2Vec(sentences, vector_size=200, window=5)
    word2vec.save(os.path.join(data_folder, 'word2vec_model'))

def load_word2vec_embeddings2(data_folder):

    word2vec_file = os.path.join(data_folder, 'word2vec_model')
    model = KeyedVectors.load(word2vec_file)
    wv = model.wv
    del model

    # embedding matrix is orderd by indices in model.wv.voacab
    word_map = {token: token_index for token_index, token in enumerate(wv.index_to_key2)}

    # embedding = np.load(w2v_word_vectors_path)
    embedding = wv.vectors
    unknown_vector = np.mean(embedding, axis=0)
    padding_vector = np.zeros(len(embedding[0]))

    embedding = np.append(embedding, unknown_vector.reshape((1, -1)), axis=0)
    embedding = np.append(embedding, padding_vector.reshape((1, -1)), axis=0)

    word_map['<UNK>'] = len(embedding) - 2  # map unknown words to vector we just appended
    word_map['<PAD>'] = len(embedding) - 1

    return embedding, word_map

class GRNNDataset2(Dataset):
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        split = split.lower()
        assert split in {'train', 'val', 'test'}
        self.split = split

        # Load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

        self.classes = sorted([int(y) for y in set(self.data['labels'])])
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, i):
        return self.data['docs'][i], self.classes.index(int(self.data['labels'][i]))

class CustomDataloader2:
    """
    On iteration it yields batches with desired batch size, generated using the given sampler.
    These batches can then again be iterated to receive tuples of document and label.
    """
    def __init__(self, batch_size, dataset, data_size):
        self._batch_size = batch_size
        self.indices = [*range(0, data_size, 1)]
        self._sampler = sampler.SubsetRandomSampler(self.indices)
        self._num_batches = len(self._sampler) // batch_size
        self._dataset = dataset

    def __iter__(self):
        i = 0
        batch_idx = []
        for doc_idx in self._sampler:
            batch_idx.append(doc_idx)
            i += 1
            if i % self._batch_size == 0:
                yield self._batch_iterator(batch_idx)
                batch_idx = []

    def __len__(self):
        return len(self._sampler) // self._batch_size

    def _batch_iterator(self, batch: List[int]):
        for i in batch:
            (doc, label) = self._dataset[i]
            yield (doc, label)