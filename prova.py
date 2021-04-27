import pandas as pd


def readIMDB():
    dataset_name = 'imdb_reviews'
    data_df = pd.read_csv('datasets/'+dataset_name+'.csv')
    print("shape df: ", data_df.shape)

    return dataset_name, 2, data_df
