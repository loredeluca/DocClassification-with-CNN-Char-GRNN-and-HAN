import string
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical


def preprocessing(csv_folder, header):
    # TODO: se il dataset non è gia splittato in train e test --> splittalo e salvalo
    print('\n Reading training data...\n')
    (train_docs, train_labels) = read_csv(csv_folder, 'train', header)
    print('\n Reading test data...\n')
    (test_docs, test_labels) = read_csv(csv_folder, 'test', header)

    # Create alphabet/vocab
    print('\n Creating vocab...\n')
    # l'alfabeto comprende lettere, numeri, punteggiatura e \n.
    # Nel paper dice che sono 70 caratteri perche include due simboli '-', ma in realtà sono 69
    # Vedi https://github.com/zhangxiangxiao/Crepe#issues.
    alphabet = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + ['\n'])
    alphabet_size = len(alphabet)
    check = set(alphabet)  # rende la lista iterabile

    vocab = {}
    reverse_vocab = {}
    for idx, char in enumerate(alphabet):
        vocab[char] = idx
        reverse_vocab[idx] = char
    # creo un vocab che ha come key il carattere e uno che ha come key l'indice
    # vocab = {'a': 0, 'b': 1, 'c': 2, ..}
    # reverse_vocab = {0: 'a', 1: 'b', 2: 'c', ..}

    """
    # Encode
    print("Encoding test data...\n")
    # Crea una matrice maxlen x alphabet_size (in questo caso 256(1014)x69)
    # e si crea una matrice 3D data x maxlen x alphabet_size
    # ogni carattere subisce la 'one-hot encode'
    # (caratteri non presenti nell'alphabet codificati con vettore tutto zero)

    maxlen = 256 #indica frase di len max = 256 caratteri

    #encoded_train_docs =

    encoded_test_docs = encode_data(test_docs, maxlen, vocab, alphabet_size, check)
    """

    return (train_docs, train_labels), (test_docs, test_labels), vocab, alphabet_size, check


def read_csv(csv_folder, split, header):

    assert split in {'train', 'test'}

    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'), header=0 if header is True else None)
    #  elimino eventuali righe vuote
    data = data.dropna()

    if header:
        x_data = np.array(data['text'])
        y_data = to_categorical(data['label']-1)

    else:
        # se non ha l'header:
        # seleziona la/le colonna/e in cui c'è il text
        x_data = np.array(data[1]+data[2])
        # seleziona la colonna in cui c'è la label:
        # se parte da 1, la faccio partire da 0 e con to_categorical faccio 'one hot encoding'
        y_data = to_categorical(data[0]-1)

    return x_data, y_data

