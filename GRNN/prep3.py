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
from torch.utils.data import Dataset, sampler
from typing import List
import pickle


class GRNNDataset(Dataset):
    def __init__(self, csv_folder, output_folder):

        self._x_data, self._y_data, self.embedding, self.word2index = self.GRNN_prep2(csv_folder, output_folder)
        self.index2word = {index: word for (word, index) in self.word2index.items()}

        self.classes = sorted([int(y) for y in set(self._y_data)])
        self.num_classes = len(self.classes)


    def __len__(self):
        return len(self._x_data)

    def __getitem__(self, i):
        return self._x_data[i], self.classes.index(int(self._y_data[i]))

    def GRNN_prep2(self, csv_folder, output_folder): #, sentence_limit, word_limit, save_word2vec_data=True):
        print('Reading and tokenize data...\n')

        x_data_prep, y_data, n_classes = self.read_csv2(csv_folder)

        # Save text data for word2vec
        # if save_word2vec_data:
        #    torch.save(train_texts, os.path.join(output_folder, 'word2vec_data.pth.tar'))
        #    print('\nText data for word2vec saved to %s.\n' % os.path.abspath(output_folder))

        print('\n Train word2vec model...')
        self.train_word2vec_model2(x_data_prep, output_folder)
        print('\nEND TRAINING WORD2VEC MODEL\n')

        print("Loading word2vec model...")
        embedding, word2index = self.load_word2vec_embeddings2(output_folder)
        x_data_index = self.words_to_vocab_index2(x_data_prep, word2index)

        # Save
        with open(os.path.join(output_folder, f"x_data_text"), "wb") as savefile:
            pickle.dump(x_data_prep, savefile)
        with open(os.path.join(output_folder, f"x_data"), "wb") as savefile:
            pickle.dump(x_data_index, savefile)
        with open(os.path.join(output_folder, f"y_data"), "wb") as savefile:
            pickle.dump(y_data, savefile)

        print("END PREPROCESSING!\n")

        return x_data_index, y_data, embedding, word2index  # embedding, word2index, index2word, n_classes

    def read_csv2(self, data_folder):

        data = pd.read_csv(os.path.join(data_folder, '.csv'))  # , header=None)

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

    def train_word2vec_model2(self, data_folder, docs):
        # Read data
        # docs = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))
        sentences = [sentence for doc in docs for sentence in doc]

        # print intermediate info
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # Initialize and train the model (this will take some time)
        word2vec = Word2Vec(sentences=sentences, vector_size=200, window=5)

        word2vec.wv.save(os.path.join(data_folder, 'word2vec_model'))

    def load_word2vec_embeddings2(self, data_folder):
        # Load word2vec model into memory
        word2vec_file = os.path.join(data_folder, 'word2vec_model')
        w2v = KeyedVectors.load(word2vec_file)

        # embedding matrix is orderd by indices in model.wv.voacab
        word2index = {token: token_index for token_index, token in enumerate(w2v.index_to_key)}

        # embedding = np.load(w2v_word_vectors_path)
        embedding = w2v.vectors
        unknown_vector = np.mean(embedding, axis=0)
        padding_vector = np.zeros(len(embedding[0]))

        embedding = np.append(embedding, unknown_vector.reshape((1, -1)), axis=0)
        embedding = np.append(embedding, padding_vector.reshape((1, -1)), axis=0)

        word2index['<unk>'] = len(embedding) - 2  # map unknown words to vector we just appended
        word2index['<pad>'] = len(embedding) - 1

        return embedding, word2index

    def words_to_vocab_index2(self, documents, word2index):
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




class CustomDataloader:
    """
    On iteration it yields batches with desired batch size, generated using the given sampler.
    These batches can then again be iterated to receive tuples of document and label.
    """
    def __init__(self, dataset: Dataset, sampler: sampler.Sampler, batch_size: int):
        self._batch_size = batch_size
        self._sampler = sampler
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
