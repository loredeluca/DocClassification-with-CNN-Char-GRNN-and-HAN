import numpy as np
import sys
import csv
from torch.utils.data import Dataset
import string

csv.field_size_limit(sys.maxsize)


class MyDataset(Dataset):
    def __init__(self, data_path, max_length=1014):
        self.data_path = data_path

        # l'alfabeto comprende lettere, numeri, punteggiatura e \n.
        # Nel paper dice che sono 70 caratteri perche include due simboli '-', ma in realtà sono 69
        # Vedi https://github.com/zhangxiangxiao/Crepe#issues.
        self.vocabulary = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + ['\n'])
        #self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx.lower()  # add lower
                    text += " "
                # if len(line) == 3:
                #     text = "{} {}".format(line[1].lower(), line[2].lower())
                # else:
                #     text = "{}".format(line[1].lower())
                label = int(line[0]) - 1
                texts.append(text)
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        # data è una matrice di dimensione num_caratteri_frase x vocab_size
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



max_len = 1014
training_set = MyDataset("./datasets/ag_news_csv/train.csv", max_len)

