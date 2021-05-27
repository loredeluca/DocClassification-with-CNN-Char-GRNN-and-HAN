from torch.utils.data import Dataset
import string
import numpy as np
import csv
import os
#import sys

#csv.field_size_limit(sys.maxsize)


class CN_ChartDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
    """
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # L'alfabeto comprende lettere, numeri, punteggiatura e \n:
        # nel paper 70 caratteri (include due simboli '-', qui 69 caratteri
        self.vocabulary = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + ['\n'])
        self.identity_mat = np.identity(len(self.vocabulary))

        texts, labels = [], []
        with open(os.path.join(data_folder, split+'.csv')) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx.lower()  # add lower
                    text += " "
                label = int(line[0]) - 1
                texts.append(text)
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = 1014 # max length of characters in a sentence
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        # data è una matrice (num_caratteri_frase x vocab_size)
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        # se la lunghezza della frase > max_length --> la taglio
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label



#max_len = 1014
#training_set = CN_ChartDataset("./datasets/ag_news_csv/train.csv", max_len)

