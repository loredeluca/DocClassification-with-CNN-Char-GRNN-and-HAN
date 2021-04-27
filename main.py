from prova import readIMDB

if __name__ == '__main__':

    dataset_name, n_classes, data_df = readIMDB()
    print(data_df.head())


