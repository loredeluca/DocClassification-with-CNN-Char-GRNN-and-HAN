import numpy as np
import torch
from torch import nn
import os
from itertools import chain
import logging
from gensim.models import Word2Vec, KeyedVectors


def save_checkpoint2(type_, epoch, model, optimizer, word_map=None):
    state, filename = dict, str
    if type_ == 'cnn-chart':
        state = {'epoch': epoch,
                 'model': model,
                 'optimizer': optimizer,
                 'word_map': word_map}
        filename = 'checkpoint_CN-Char.pth.tar'
    elif type_ == 'grnn':
        state = {'epoch': epoch,
                 'model': model,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()}
        filename = 'checkpoint_GRNN.pth.tar'
    elif type_ == 'han':
        state = {'epoch': epoch,
                 'model': model,
                 'optimizer': optimizer}
        filename = 'checkpoint_CN-han.pth.tar'

    torch.save(state, filename)


def train_word2vec_model(data_folder, model):
    # Train a word2vec model for word embeddings.

    # Read data
    if model == 'han':
        sentences = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))
        sentences = list(chain.from_iterable(sentences))
    elif model == 'grnn':
        docs = torch.load(os.path.join(data_folder, 'word2vec_data_grnn.pth.tar'))
        sentences = [sentence for doc in docs for sentence in doc]

    # print intermediate info
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize and train the model (this will take some time)
    word2vec = Word2Vec(sentences=sentences, vector_size=200, workers=8, window=10, min_count=5)

    # Normalize vectors and save model
    word2vec.init_sims(True)
    word2vec.wv.save(os.path.join(data_folder, 'word2vec_model'))


def load_word2vec_embeddings_han(word2vec_file, word_map):
    # Load pre-trained embeddings for words in the word map.
    w2v = KeyedVectors.load(word2vec_file, mmap='r')
    embedding_size = w2v.vector_size

    print("\nEmbedding length is %d.\n" % embedding_size)

    # Create tensor to hold embeddings for words that are in-corpus
    embeddings = torch.FloatTensor(len(word_map), embedding_size)

    bias = np.sqrt(3.0 / embeddings.size(1))
    nn.init.uniform_(embeddings, -bias, bias)

    # Read embedding file
    print("Loading embeddings...")
    for word in word_map:
        if word in w2v.key_to_index:
            embeddings[word_map[word]] = torch.FloatTensor(w2v[word])

    print("Done.\n Embedding vocabulary: %d.\n" % len(word_map))

    return embeddings, embedding_size

def load_word2vec_embeddings_grnn(data_folder):

    word2vec_file = os.path.join(data_folder, 'word2vec_model')
    w2v = KeyedVectors.load(word2vec_file)
    wv = w2v.wv
    del w2v

    # embedding matrix is orderd by indices in wv.vocab
    word_map = {token: token_index for token_index, token in enumerate(wv.index_to_key)}

    embedding = wv.vectors
    unknown_vector = np.mean(embedding, axis=0)
    padding_vector = np.zeros(len(embedding[0]))

    embedding = np.append(embedding, unknown_vector.reshape((1, -1)), axis=0)
    embedding = np.append(embedding, padding_vector.reshape((1, -1)), axis=0)

    word_map['<UNK>'] = len(embedding) - 2
    word_map['<PAD>'] = len(embedding) - 1

    return embedding, word_map


def adjust_learning_rate(optimizer, scale_factor):
    print("\n Decaying learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor

