from torch.utils.data import Dataset
import pandas as pd
import os
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
import numpy as np
import torch
import json
from itertools import chain
import logging
from gensim.models import Word2Vec


class HANDataset(Dataset):
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

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), torch.LongTensor([self.data['labels'][i]])


def HAN_preprocessing(csv_folder, output_folder, sentence_limit, word_limit, min_word_count=5, save_word2vec_data=True):
    print('Reading and preprocessing training data...\n')
    train_texts, train_labels, word_counter, n_classes = read_csv(csv_folder, 'train', sentence_limit, word_limit)

    # Save text data for word2vec
    if save_word2vec_data:
        torch.save(train_texts, os.path.join(output_folder, 'word2vec_data.pth.tar'))
        print('\nText data for word2vec saved to %s.\n' % os.path.abspath(output_folder))

    # Create word map (=build vocabulary, remove unique words)
    # set <pad>=0 e <unk>=last_word --> word_map={'<pad>': 0, 'parola': 1,.., 'parola':n-1, '<unk>': n}
    # ==> word_map = vocabolario =word_voc <==
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    print('Discarding words with counts less than %d, the size of the vocabulary is %d.\n' % (min_word_count, len(word_map)))

    with open(os.path.join(output_folder, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)
    print('Word map saved to %s.\n' % os.path.abspath(output_folder))

    # Encode and pad
    # NB: x = lambda a: a + 10  equivale a  def x(a):
    #       	                              return a+10
    print('Encoding and padding training data...\n')
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, 0), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), train_texts))
    sentences_per_train_document = list(map(lambda doc: len(doc), train_texts))
    words_per_train_sentence = list(map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), train_texts))

    # Save
    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(sentences_per_train_document) == len(words_per_train_sentence)
    torch.save({'docs': encoded_train_docs,
                'labels': train_labels,
                'sentences_per_document': sentences_per_train_document,
                'words_per_sentence': words_per_train_sentence},
               os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Encoded, padded training data saved to %s.\n' % os.path.abspath(output_folder))
    # Free some memory
    del train_texts, encoded_train_docs, train_labels, sentences_per_train_document, words_per_train_sentence

    #Read and preprocessing VALIDATION data
    data_validation(csv_folder, output_folder, sentence_limit, word_limit, word_map)

    # Read and preprocessing TEST data
    print('Reading and preprocessing test data...\n')
    test_docs, test_labels, _, _ = read_csv(csv_folder, 'test', sentence_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding test data...\n')
    encoded_test_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, 0), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), test_docs))
    sentences_per_test_document = list(map(lambda doc: len(doc), test_docs))
    words_per_test_sentence = list(map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), test_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(sentences_per_test_document) == len(words_per_test_sentence)
    torch.save({'docs': encoded_test_docs,
                'labels': test_labels,
                'sentences_per_document': sentences_per_test_document,
                'words_per_sentence': words_per_test_sentence},
               os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Encoded, padded test data saved to %s.\n' % os.path.abspath(output_folder))

    print('END PREPROCESSING!\n')

    print('\n Train word2vec model...')
    train_word2vec_model(data_folder=output_folder)
    print('\nEND TRAINING WORD2VEC MODEL\n')

    return word_map, n_classes


def read_csv(data_folder, split, sentence_limit, word_limit):

    split = split.lower()
    assert split in {'train', 'test'}

    # Tokenizers
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()

    """
    texts, labels = [], []
    with open(os.path.join(data_folder, split+'.csv')) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                sentences = list()
                for tx in line[2:]:
                    for paragraph in tx.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n').splitlines():
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
                if row[1] == 'not_relevant':
                    row[1] = 1

                # Faccio partire le classi da 0 invece che da 1
                labels.append(int(row[1]) - 1)
                docs.append(words)        
    """
    data = pd.read_csv(os.path.join(data_folder, split + '.csv'))  # , header=None)
    docs, labels = [], []
    word_counter = Counter()
    for i in tqdm(range(data.shape[0])):
        # For each row in the date, insert the element of each column into 'line'
        # es: line = [7, ciao come stai, user120, 21-12-20,..]
        line = list(data.loc[i, :])

        sentences = list()
        # divide text and label: (line[0]=indice, line[1]=label, line[2:]=testo)
        for text in line[2:]: #sarebbe row[1:0]
            if isinstance(text, float):
                return ''
            # clean and split
            # es: ['ciao come stai? sto bene] -> ['ciao come stai?', 'sto bene']
            for paragraph in text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n').splitlines():
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
    word2vec = Word2Vec(sentences=sentences, vector_size=200, workers=8, window=10, min_count=5)

    # Normalize vectors and save model
    word2vec.init_sims(True)
    word2vec.wv.save(os.path.join(data_folder, 'word2vec_model'))



def data_validation(csv_folder, output_folder, sentence_limit, word_limit, word_map):
    # Read and preprocessing VALIDATION data
    print('Reading and preprocessing validation data...\n')  # test or validation
    eval_docs, eval_labels, _, _ = read_csv(csv_folder, 'val', sentence_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding val data...\n')
    encoded_eval_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, 0), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), eval_docs))
    sentences_per_eval_document = list(map(lambda doc: len(doc), eval_docs))
    words_per_eval_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), eval_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_eval_docs) == len(eval_labels) == len(sentences_per_eval_document) == len(
        words_per_eval_sentence)
    torch.save({'docs': encoded_eval_docs,
                'labels': eval_labels,
                'sentences_per_document': sentences_per_eval_document,
                'words_per_sentence': words_per_eval_sentence},
               os.path.join(output_folder, 'VAL_data.pth.tar'))
    print('Encoded, padded test data saved to %s.\n' % os.path.abspath(output_folder))

