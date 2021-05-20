from collections import Counter
import pandas as pd
import os
from tqdm import tqdm
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
import torch
import json
import itertools
import logging
import gensim
import numpy as np

def preprocessing(csv_folder, output_folder, sentence_limit, word_limit, min_word_count=5, save_word2vec_data=True):
    # Read e little preprocessing training data
    print('\nReading and preprocessing training data...\n')
    train_docs, train_labels, word_counter, n_classes = read_csv(csv_folder, 'train', sentence_limit, word_limit)

    # Save text data for word2vec
    if save_word2vec_data:
        torch.save(train_docs, os.path.join(output_folder, 'word2vec_data.pth.tar'))
        print('\nText data for word2vec saved to %s.\n' % os.path.abspath(output_folder))

    # Create word map
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    print('\nDiscarding words with counts less than %d, the size of the vocabulary is %d.\n' % (min_word_count, len(word_map)))

    with open(os.path.join(output_folder, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)
    print('Word map saved to %s.\n' % os.path.abspath(output_folder))

    # Encode and pad
    print('Encoding and padding training data...\n')
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), train_docs))
    sentences_per_train_document = list(map(lambda doc: len(doc), train_docs))
    words_per_train_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), train_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(sentences_per_train_document) == len(
        words_per_train_sentence)
    # Because of the large data, saving as a JSON can be very slow
    torch.save({'docs': encoded_train_docs,
                'labels': train_labels,
                'sentences_per_document': sentences_per_train_document,
                'words_per_sentence': words_per_train_sentence},
               os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Encoded, padded training data saved to %s.\n' % os.path.abspath(output_folder))

    return n_classes



def preprocess(text):
    """
    Pre-process text for use in the model. This includes lower-casing, standardizing newlines, removing junk.

    :param text: a string
    :return: cleaner string
    """
    if isinstance(text, float):
        return ''

    return text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')


def read_csv(csv_folder, split, sentence_limit, word_limit):
    """
    Read CSVs containing raw training data, clean documents and labels, and do a word-count.

    :param csv_folder: folder containing the CSV
    :param split: train or test CSV?
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :return: documents, labels, a word-count
    """
    #contorolla che split sia train o test
    assert split in {'train', 'test'}

    docs = []
    labels = []
    word_counter = Counter()

    # Tokenizers
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()

    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'), encoding='latin-1')#, header=None)
    print(data)
    # per ogni riga
    for i in tqdm(range(data.shape[0])):
        # di ogni riga leggi tutti gli elementi(=tutte le colonne) e li mette in una lista
        # es: row = [7, ciao come stai, user120, 21-12-20,..]
        row = list(data.loc[i, :])

        sentences = list()
        # per tutta la roba
        for text in row[2:]: #sarebbe row[1:0]

            # fa la preelaborazione del testo
            for paragraph in preprocess(text).splitlines():
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

        for x in row[1]:
            if row[1] == 'not_relevant':
                row[1] = 1

        labels.append(int(row[1]) - 1)  # since labels are 1-indexed in the CSV (lo fa partire da 0?)
        docs.append(words)
    n_classes = len(np.unique(labels))
    print("docs: ", docs)
    print("label: ", labels)
    print("word_C", word_counter)
    print('n_classes', n_classes)
    #ritorna docs = text, tokenizzato parola per parola [['ciao','come,'sta'],[..]]
    # labels = labels da 0 a n-1
    #word_counter = ['ciao': 12, 'casa':  8, ...]



    return docs, labels, word_counter, n_classes

def train_word2vec_model(data_folder):#, algorithm='skipgram'):
    """
    Train a word2vec model for word embeddings.

    See the paper by Mikolov et. al. for details - https://arxiv.org/pdf/1310.4546.pdf

    :param data_folder: folder with the word2vec training data
    :param algorithm: use the Skip-gram or Continous Bag Of Words (CBOW) algorithm?
    """
    #assert algorithm in ['skipgram', 'cbow']
    #sg = 1 if algorithm is 'skipgram' else 0

    # Read data
    sentences = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))
    sentences = list(itertools.chain.from_iterable(sentences))

    # Activate logging for verbose training
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize and train the model (this will take some time)
    model = gensim.models.word2vec.Word2Vec(sentences=sentences, vector_size=200, workers=8, window=10, min_count=5)#, sg=sg)

    # Normalize vectors and save model
    model.init_sims(True)
    model.wv.save(os.path.join(data_folder, 'word2vec_model'))


preprocessing(csv_folder='./datasets', output_folder='./han_data', sentence_limit=15, word_limit=20, min_word_count=5)