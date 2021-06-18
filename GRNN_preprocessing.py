from torch.utils.data import Dataset, sampler
import torch
import os
import pandas as pd
from tqdm import tqdm
import re
import string
from gensim.utils import simple_preprocess

from utils import train_word2vec_model, load_word2vec_embeddings_grnn


class GRNNDataset(Dataset):
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

        self.classes = sorted([int(y) for y in set(self.data['labels'])])
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, i):
        return self.data['docs'][i], self.classes.index(int(self.data['labels'][i]))


def GRNN_preprocess(csv_folder, output_folder, save_word2vec_data=True):
    # Read and Preprocessing train data
    print('Reading and preprocessing train_data...\n')
    train_texts, train_labels, train_size = prep_data(csv_folder, 'train')

    # Save text data for word2vec
    if save_word2vec_data:
        torch.save(train_texts, os.path.join(output_folder, 'word2vec_data_grnn.pth.tar'))

    # Read and Preprocessing val data
    print('Reading and preprocessing val_data...\n')
    val_texts, val_labels, val_size = prep_data(csv_folder, 'val')

    # Read and Preprocessing test data
    print('Reading and preprocessing test_data...\n')
    test_texts, test_labels, test_size = prep_data(csv_folder, 'test')

    print('\n Train word2vec model...')
    train_word2vec_model(data_folder=output_folder, model='grnn')
    print('\nEND TRAINING WORD2VEC MODEL\n')

    # Build word_map & embedding
    embedding, word_map = load_word2vec_embeddings_grnn(output_folder)

    # Encode train data
    encode_data('train', train_texts, word_map, train_labels, output_folder)
    # Encode val data
    encode_data('val', val_texts, word_map, val_labels, output_folder)
    # Encode test data
    encode_data('test', test_texts, word_map, test_labels, output_folder)

    print('END PREPROCESSING!\n')
    return embedding, word_map, train_size, val_size, test_size


def prep_data(data_folder, split):
    data = pd.read_csv(os.path.join(data_folder, split + '.csv'))
    data_size = data.shape[0]

    y_data = data['label']
    x_data = data['text']

    x_data_prep = []
    for i, doc in enumerate(tqdm(x_data)):
        x_data_prep.append([])
        split = re.split('\!|\.|\?|\;|\:|\n', doc)
        for sentence in split:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            x_data_prep[-1].append(simple_preprocess(sentence, min_len=1, max_len=20, deacc=True))

    return x_data_prep, y_data, data_size


def encode_data(split, documents, word_map, labels, output_folder):
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


class CustomDataloader:
    def __init__(self, dataset, data_size):
        self.batch_size = 64  # 50
        self.indices = [*range(0, data_size, 1)]
        self.sampler = sampler.SubsetRandomSampler(self.indices)
        self.num_batches = len(self.sampler) // self.batch_size
        self.dataset = dataset

    def __iter__(self):
        i = 0
        batch_idx = []
        for doc_idx in self.sampler:
            batch_idx.append(doc_idx)
            i += 1
            if i % self.batch_size == 0:
                yield self.batch_iterator(batch_idx)
                batch_idx = []

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def batch_iterator(self, batch):
        for i in batch:
            (doc, label) = self.dataset[i]
            yield (doc, label)
