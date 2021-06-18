from torch.utils.data import Dataset
import string
import numpy as np
import csv
import os


class CN_CharDataset(Dataset):
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        split = split.lower()
        assert split in {'train', 'test'}
        self.split = split

        # The alphabet includes letters, numbers, punctuation and \ n:
        # paper alphabet: 70 characters (includes two '-' symbols), my alphabet: 69 characters
        self.alphabet = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + ['\n'])
        self.identity_mat = np.identity(len(self.alphabet))

        texts, labels = [], []
        with open(os.path.join(data_folder, split+'.csv')) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            # if the dataset has the header, not read idx=0
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx.lower()
                    text += " "
                label = int(line[0]) - 1
                texts.append(text)
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = 1014  # max length of characters in a sentence
        self.length = len(self.labels)
        self.n_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        # dim_data: (num_characters_sent x vocab_size)
        data = np.array([self.identity_mat[self.alphabet.index(i)] for i in list(raw_text) if i in self.alphabet],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.alphabet)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.alphabet)), dtype=np.float32)
        label = self.labels[index]
        return data, label
