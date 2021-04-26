from prova import getIMDB

if __name__ == '__main__':

    dataset_name, n_classes, data_df = getIMDB()
    print(dataset_name, n_classes)
    print(data_df.head())
