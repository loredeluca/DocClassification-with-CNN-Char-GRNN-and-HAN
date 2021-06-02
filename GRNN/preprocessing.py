import numpy as np
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter
import torch
from itertools import chain
import logging
from gensim.models import Word2Vec, KeyedVectors

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


def GRNN_prep(csv_folder, output_folder, sentence_limit, word_limit, save_word2vec_data=True):
    print('Reading and preprocessing training data...\n')
    # TRAIN
    train_texts, train_labels, word_counter, n_classes = read_csv(csv_folder, 'train', sentence_limit, word_limit)

    # Save text data for word2vec
    if save_word2vec_data:
        torch.save(train_texts, os.path.join(output_folder, 'word2vec_data.pth.tar'))
        print('\nText data for word2vec saved to %s.\n' % os.path.abspath(output_folder))

    # Save
    print('Saving data train...\n')
    torch.save({'docs': train_texts, 'labels': train_labels}, os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Training data saved to %s.\n' % os.path.abspath(output_folder))

    # VAL
    eval_texts, eval_labels, _, _ = read_csv(csv_folder, 'val', sentence_limit, word_limit)
    # Save
    print('Saving...\n')
    torch.save({'docs': eval_texts, 'labels': eval_labels}, os.path.join(output_folder, 'VAL_data.pth.tar'))
    print('Val data saved to %s.\n' % os.path.abspath(output_folder))

    # VAL
    test_texts, test_labels, _, _ = read_csv(csv_folder, 'test', sentence_limit, word_limit)
    # Save
    print('Saving...\n')
    torch.save({'docs': test_texts, 'labels': test_labels}, os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Test data saved to %s.\n' % os.path.abspath(output_folder))

    print("END PREPROCESSING!\n")


    print('\n Train word2vec model...')
    train_word2vec_model(data_folder=output_folder)
    print('\nEND TRAINING WORD2VEC MODEL\n')

    print("Loading word2vec model...")
    embedding, word2index = load_word2vec_embeddings(output_folder)
    words_to_vocab_index(train_texts, word2index)

    return train_texts, train_labels, embedding, word2index, n_classes


def read_csv(data_folder, split, sentence_limit, word_limit):
    split = split.lower()
    assert split in {'train', 'val', 'test'}

    # Tokenizers
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()

    data = pd.read_csv(os.path.join(data_folder, split + '.csv'))  # , header=None)
    data.head()
    docs, labels = [], []
    word_counter = Counter()
    for i in tqdm(range(data.shape[0])):
        # For each row in the date, insert the element of each column into 'line'
        # es: line = [7, ciao come stai, user120, 21-12-20,..]
        line = list(data.loc[i, :])

        sentences = list()
        # divide text and label: (line[0]=indice, line[1]=label, line[2:]=testo)
        for text in line[2:]:  # sarebbe row[1:0]
            if isinstance(text, float):
                return ''
            # clean and split
            # es: ['ciao come stai? sto bene] -> ['ciao come stai?', 'sto bene']
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
        if line[1] == 'not_relevant':
            line[1] = 1

        labels.append(int(line[1]) - 1)
        docs.append(words)

    n_classes = len(np.unique(labels))
    # print("docs: ", docs)
    # print("label: ", labels)
    # print("word_C", word_counter)
    # print('n_classes', n_classes)
    # docs = text, tokenizzato parola per parola [['ciao','come,'sta'],[..]]
    # labels = labels da 0 a n-1
    # word_counter = ['ciao': 12, 'casa':  8, ...]

    return docs, labels, word_counter, n_classes


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
    word2vec = Word2Vec(sentences=sentences, vector_size=200, window=5)

    word2vec.wv.save(os.path.join(data_folder, 'word2vec_model'))


def load_word2vec_embeddings(data_folder):
    # Load word2vec model into memory
    word2vec_file = os.path.join(data_folder, 'word2vec_model')
    w2v = KeyedVectors.load(word2vec_file, mmap='r')

    # embedding matrix is orderd by indices in model.wv.voacab
    word2index = {token: token_index for token_index, token in enumerate(w2v.wv.index2word)}

    # embedding = np.load(w2v_word_vectors_path)
    embedding = w2v.wv.vectors
    unknown_vector = np.mean(embedding, axis=0)
    padding_vector = np.zeros(len(embedding[0]))

    embedding = np.append(embedding, unknown_vector.reshape((1, -1)), axis=0)
    embedding = np.append(embedding, padding_vector.reshape((1, -1)), axis=0)

    word2index['<unk>'] = len(embedding) - 2  # map unknown words to vector we just appended
    word2index['<pad>'] = len(embedding) - 1

    return embedding, word2index


def words_to_vocab_index(documents, word2index):
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
