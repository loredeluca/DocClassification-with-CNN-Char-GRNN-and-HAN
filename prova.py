import tensorflow_datasets as tfds
import pandas as pd
import numpy as np


def getIMDB():
    dataset_name = 'imdb_reviews'
    # dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True, shuffle_files=True)
    ds = tfds.load(dataset_name, split='train')
    reviews = []
    for element in ds.as_numpy_iterator():
        reviews.append((element['text'].decode('utf-8'), element['label']))
        print((element['text'].decode('utf-8'), element['label']))

    data_df = pd.DataFrame(data=reviews, columns=['text', 'label'])

    return dataset_name, 2, data_df