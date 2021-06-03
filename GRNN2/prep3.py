from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import pandas as pd
import string
import re
from typing import List
import gensim as gs


class YelpDataset(Dataset):

    def __init__(self, data_paths, name, w2v_path, prep_path, w2v_model_name = None,
                 overwrite: bool = False, embedding_dim: int = 200, w2v_sample_frac: float = 0.3):
        """
        Load a given YELP rating dataset. If <overwrite> is true or there is no persisted data for the given <_name> yet,
        the data will be preprocessed and the results persisted under the given paths <w2v_path> and <prep_path>.

        :param data_paths: One or multiple paths to raw data files to load
        :param name: Name of the data to load (for naming output files)
        :param overwrite: If there are files with the given _name already, rebuild model and overwrite them or load them?
        :param w2v_path: Path to Word2Vec directory (for persistence)
        :param prep_path: Path to data preprocessing directory (for persistence)
        """
        self._data_paths = data_paths
        self._prep_path = prep_path
        self._name = name
        if w2v_model_name:
            self._w2v_model_name = w2v_model_name
        else:
            self._w2v_model_name = name
        self._overwrite = overwrite

        self._embedding_dim = embedding_dim
        self._w2v_sample_frac = w2v_sample_frac
        self._w2v = Word2Vector(data_paths, w2v_path, self._w2v_model_name, self._overwrite, self._embedding_dim)

        # if name = 'imdb':
        #   self._imdb_rating_index = 2
        #   self._imdb_review_index = 3
        # else:
        self._yelp_rating_key = 'label'
        self._yelp_review_key = 'text'

        self._X_data, self._y_data, self.embedding, self.word2index = self._load()
        self.index2word = {index: word for (word, index) in self.word2index.items()}
        self.classes = sorted([int(y) for y in set(self._y_data)])

        self.num_classes = len(self.classes)

        self.unknown_word_key = self._w2v.unknown_word_key
        self.padding_word_key = self._w2v.padding_word_key

    def get_class_distr(self, labels):
        class_distr = np.zeros((len(self.classes),))
        for y in labels:
            y = int(y)
            class_distr[y - 1] += 1
        return class_distr

    def __getitem__(self, index):
        """
        Get tuple of data and label, where the label is the index of the class in YelpDataset.classes.
        """
        return self._X_data[index], self.classes.index(int(self._y_data[index]))

    def __len__(self):
        return len(self._X_data)

    def _load(self):
        """
        Preprocess Yelp data: Extract text and rating data and replace words by vocabulary ids.
        :return: List of documents with vocabulary indices instead of words, list of ratings and word embedding matrix
        """
        if self._overwrite or \
                (not os.path.isfile(self._X_path())) or \
                (not os.path.isfile(self._y_path())):
            print("No persisted data found. Preprocessing data...")
            X_data, y_data, embedding, word2index = self._preprocess()
        else:
            print("Persisted data found. Loading...")
            X_data, y_data, embedding, word2index = self._load_preprocessed()

        return X_data, y_data, embedding, word2index

    def _load_preprocessed(self):
        with open(self._X_path(), "rb") as file:
            X_data = pickle.load(file)
        with open(self._y_path(), "rb") as file:
            y_data = pickle.load(file)
        embedding, word2index = self._w2v.get_embedding(X_data)
        return X_data, y_data, embedding, word2index

    def _preprocess(self):
        # load data and extract text information as X and rating as y:
        data = pd.DataFrame()
        if type(self._data_paths) == List[str]:
            for path in self._data_paths:
                data_in = pd.read_json(path, lines=True)
                data = pd.concat([data, data_in], axis=0)
                del data_in
        else:
            data = pd.read_csv(self._data_paths)
            # TODO: per imdb cambia qualcosa
        # y is a list of gold star ratings for reviews
        y_data = data[self._yelp_rating_key]
        # X is a list with all documents, where documents are lists of sentences and each sentence-list
        # contains single words as strings
        X_data_text = data[self._yelp_review_key]
        # Separate and preprocess words in sentences
        X_data_prep = []
        for i, doc in enumerate(X_data_text):
            if i % 1000 == 0:
                print(f"Processing documents {i} - {i+999} of {len(X_data_text)}...")
            X_data_prep.append([])
            split = re.split('\!|\.|\?|\;|\:|\n', doc)
            for sentence in split:
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                X_data_prep[-1].append(gs.utils.simple_preprocess(sentence, min_len=1, max_len=20, deacc=True))
        print("Building word2vec model...")
        embedding, word2index = self._w2v.get_embedding(X_data_prep)
        X_data_index = self._words_to_vocab_index(X_data_prep, word2index)
        with open(self._X_text_path(), "wb") as savefile:
            pickle.dump(X_data_prep, savefile)
        with open(self._X_path(), "wb") as savefile:
            pickle.dump(X_data_index, savefile)
        with open(self._y_path(), "wb") as savefile:
            pickle.dump(y_data, savefile)
        return X_data_index, y_data, embedding, word2index

    def _words_to_vocab_index(self, documents, word2index):
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
                        sentence_ind.append(word2index[self._w2v.unknown_word_key])
        return documents_ind

    def _y_path(self):
        return os.path.join(self._prep_path, f"y_{self._name}")

    def _X_text_path(self):
        return os.path.join(self._prep_path, f"X_{self._name}_text")

    def _X_path(self):
        return os.path.join(self._prep_path, f"X_{self._name}")


class CustomDataloader:
    """
    On iteration it yields batches with desired batch size, generated using the given sampler.
    These batches can then again be iterated to receive tuples of document and label.
    """
    def __init__(self, batch_size, sampler, dataset):
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


class Word2Vector:
    def __init__(self, data_paths, w2v_path, name, overwrite=False, dim=200):
        self._data_paths = data_paths
        self._w2v_path = w2v_path
        self._name = name
        self._overwrite = overwrite
        self._dim = dim

        self.unknown_word_key = '__UNK__'
        self.padding_word_key = '__PAD__'

    def get_embedding(self, docs):

        if self._overwrite or (not os.path.isfile(self._w2v_model_path())):
            print("No persisted word2vec model found. Creating a new embedding...")
            self._make_embedding(docs)
        print("Loading word2vec model...")
        return self._load_embedding()

    def _make_embedding(self, docs):
        sentences = [sentence for doc in docs for sentence in doc]
        model = gs.models.Word2Vec(sentences, vector_size=self._dim, window=5)
        model.save(self._w2v_model_path())

    def _load_embedding(self):
        """
        Load word embedding from the given word2vec model and extend it with vectors for unknown words and padding.
        :param w2v_model_name: Name of the word2vec model to use
        :return: Word embedding matrix and word2index mapping
        """
        if not os.path.isfile(self._w2v_model_path()):
            raise FileNotFoundError(f"Can't find a Word2Vec model with name '{self._name}' on path '{self._w2v_path}'")

        model = gs.models.KeyedVectors.load(self._w2v_model_path())
        wv = model.wv
        del model

        # embedding matrix is orderd by indices in model.wv.voacab
        word2index = {token: token_index for token_index, token in enumerate(wv.index2word)}

        # embedding = np.load(w2v_word_vectors_path)
        embedding = wv.vectors
        unknown_vector = np.mean(embedding, axis=0)
        padding_vector = np.zeros(len(embedding[0]))

        embedding = np.append(embedding, unknown_vector.reshape((1, -1)), axis=0)
        embedding = np.append(embedding, padding_vector.reshape((1, -1)), axis=0)

        word2index[self.unknown_word_key] = len(embedding) - 2  # map unknown words to vector we just appended
        word2index[self.padding_word_key] = len(embedding) - 1

        return embedding, word2index

    def _w2v_model_path(self):
        return os.path.join(self._w2v_path, f'{self._name}_w2v_model')

    def _w2v_corpus_path(self):
        return os.path.join(self._w2v_path, f'{self._name}_w2v_corpus_train')
