'''
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle
import pandas as pd
import string
import re
'''
from typing import List
from gensim.models import Word2Vec, KeyedVectors
from itertools import chain
from gensim.utils import simple_preprocess
from tqdm import tqdm
import logging
from torch.utils.data import Dataset
import pandas as pd
import os
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
import numpy as np
import torch
import json
import random


def GRNN_preprocess(csv_folder, output_folder, sentence_limit, word_limit, min_word_count=5, save_word2vec_data=True):
    # Read and Preprocessing train data
    print('Reading and preprocessing train_data...\n')
    train_texts, train_labels, word_counter, n_classes = read_csv(csv_folder, 'train', sentence_limit, word_limit)

    # Save text data for word2vec
    if save_word2vec_data:
        torch.save(train_texts, os.path.join(output_folder, 'word2vec_data.pth.tar'))

    # Build word_map (=vocabulary, remove unique words)
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)

    # Save word_map
    with open(os.path.join(output_folder, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)

    split_preprocessing('train', train_texts, train_labels, output_folder, sentence_limit, word_limit, word_map)

    # Read and Preprocessing val data
    print('Reading and preprocessing val data...\n')
    val_texts, val_labels, _, _ = read_csv(csv_folder, 'val', sentence_limit, word_limit)
    split_preprocessing('val', val_texts, val_labels, output_folder, sentence_limit, word_limit, word_map)

    # Read and Preprocessing test data
    print('Reading and preprocessing test data...\n')
    test_texts, test_labels, _, _ = read_csv(csv_folder, 'test', sentence_limit, word_limit)
    split_preprocessing('test', test_texts, test_labels, output_folder, sentence_limit, word_limit, word_map)

    print('END PREPROCESSING!\n')

    print('\n Train word2vec model...')
    train_word2vec_model(data_folder=output_folder)
    print('\nEND TRAINING WORD2VEC MODEL\n')

    '''
        embedding, word2index = self._w2v.get_embedding(X_data_prep)
        X_data_index = self._words_to_vocab_index(X_data_prep, word2index)
        with open(self._X_text_path(), "wb") as savefile:
            pickle.dump(X_data_prep, savefile)
        with open(self._X_path(), "wb") as savefile:
            pickle.dump(X_data_index, savefile)
        with open(self._y_path(), "wb") as savefile:
            pickle.dump(y_data, savefile)
        return X_data_index, y_data, embedding, word2index
    '''

    return word_map, n_classes


def read_csv(data_folder, split, sentence_limit, word_limit):
    split = split.lower()
    assert split in {'train', 'val', 'test'}

    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()

    data = pd.read_csv(os.path.join(data_folder, split + '.csv'))  # , header=None)
    docs, labels = [], []
    word_counter = Counter()
    for i in tqdm(range(data.shape[0])):
        line = list(data.loc[i, :])

        sentences = list()
        # (line[0]=indice, line[1]=label, line[2:]=testo)
        for text in line[2:]:
            if isinstance(text, float):
                return ''
            for paragraph in text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace(
                    '&#xd;', '\n').splitlines():
                sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

        words = list()
        for s in sentences[:sentence_limit]:
            w = word_tokenizer.tokenize(s)[:word_limit]
            # If sentence is empty (due to removing punctuation, digits, etc.)
            if len(w) == 0:
                continue
            words.append(w)
            word_counter.update(w)
        # If all sentences were empty
        if len(words) == 0:
            continue

        # check if all label are value
        if isinstance(line[1], str):
            line[1] = 1

        labels.append(int(line[1]) - 1)
        docs.append(words)

    n_classes = len(np.unique(labels))

    return docs, labels, word_counter, n_classes


def split_preprocessing(split, data_texts, data_labels, output_folder, sentence_limit, word_limit, word_map):
    # Encode and pad
    print('\nEncoding and padding ', split, ' data...\n')
    encoded_data_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, 0), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), data_texts))
    # Save
    print('Saving preprocessed ', split, '_texts...\n')
    torch.save({'docs': encoded_data_docs,
                'labels': data_labels},
               os.path.join(output_folder, split+'_data.pth.tar'))

class GRNNDataset(Dataset):
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

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, i):
        return self.data['docs'][i], self.data['labels'][i]
        #return torch.LongTensor(self.data['docs'][i]), torch.LongTensor([self.data['sentences_per_document'][i]]), \
        #       torch.LongTensor(self.data['words_per_sentence'][i]), torch.LongTensor([self.data['labels'][i]])

def train_word2vec_model(data_folder):
    # Sviluppiamo un word embeddings addestrando i nostri modelli word2vec su un corpus personalizzato
    # l'alternativa è usare un dataset pre-trained(Glove), in cui ad ogni parola è associata una stringa numerica
    # Train a word2vec model for word embeddings.
    # See the paper by Mikolov et. al. for details - https://arxiv.org/pdf/1310.4546.pdf

    # Read data
    sentences = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))
    sentences = list(chain.from_iterable(sentences))

    # print intermediate info
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize and train the model (this will take some time)
    word2vec = Word2Vec(sentences=sentences, vector_size=200, workers=8, window=10, min_count=5)

    # Normalize vectors and save model
    word2vec.init_sims(True)
    word2vec.wv.save(os.path.join(data_folder, 'word2vec_model'))


from torch.utils.data import sampler


class CustomDataloader:
    """
    On iteration it yields batches with desired batch size, generated using the given sampler.
    These batches can then again be iterated to receive tuples of document and label.
    """
    def __init__(self, batch_size, dataset):
        self._batch_size = batch_size
        self.indices = [*range(0, dataset.shape[0], 1)]
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


import torch.nn as nn


def load_word2vec_embeddings(word2vec_file, word_map):
    # Load pre-trained embeddings for words in the word map.
    w2v = KeyedVectors.load(word2vec_file, mmap='r')
    embedding_size = w2v.vector_size

    print("\nEmbedding length is %d.\n" % embedding_size)

    # Create tensor to hold embeddings for words that are in-corpus
    embeddings = torch.FloatTensor(len(word_map), embedding_size)

    bias = np.sqrt(3.0 / embeddings.size(1))
    nn.init.uniform_(embeddings, -bias, bias)

    # Read embedding file
    print("Loading embeddings...")
    for word in word_map:
        if word in w2v.key_to_index:
            embeddings[word_map[word]] = torch.FloatTensor(w2v[word])

    print("Done.\n Embedding vocabulary: %d.\n" % len(word_map))

    return embeddings, embedding_size