from torch.utils.data import Dataset
import pandas as pd
import os
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
import numpy as np
import torch
import json

from utils import train_word2vec_model


class HANDataset(Dataset):
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'train', 'val' or 'test'
        """
        split = split.lower()
        assert split in {'train', 'val', 'test'}
        self.split = split

        # Load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), torch.LongTensor([self.data['labels'][i]])


def HAN_preprocess(csv_folder, output_folder, sentence_limit, word_limit, min_word_count=5, save_word2vec_data=True):
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
    train_word2vec_model(data_folder=output_folder, model='han')
    print('\nEND TRAINING WORD2VEC MODEL\n')

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
    sentences_per_data_document = list(map(lambda doc: len(doc), data_texts))
    words_per_data_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), data_texts))

    # Save
    print('Saving preprocessed ', split, '_texts...\n')
    torch.save({'docs': encoded_data_docs,
                'labels': data_labels,
                'sentences_per_document': sentences_per_data_document,
                'words_per_sentence': words_per_data_sentence},
               os.path.join(output_folder, split+'_data.pth.tar'))

