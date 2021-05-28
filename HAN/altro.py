import pandas as pd

def cleanData(name):
    data_df = pd.read_csv("datasets/tweets/"+name+"1.csv", encoding='latin-1')
    data_df = data_df[["sentiment", "text"]]
    data_df.columns = ["label", "text"]
    data_df.to_csv('datasets/tweets/'+name+'.csv')

#cleanData()

def split(data_name):
    data = pd.read_csv("datasets/tweets.csv")#, encoding='latin-1')

    length = data.shape[0]

    train_len = int(0.8 * length)
    val_len = int(0.1 * length)

    train = data[:train_len]
    val = data[train_len:train_len + val_len]
    test = data[train_len + val_len:]
    #test = data[train_len:]  # train_len + val_len:]

    train.to_csv('datasets/' + data_name + '/train1.csv')
    val.to_csv('datasets/' + data_name + '/val1.csv')
    test.to_csv('datasets/' + data_name + '/test1.csv')

    cleanData('train')
    cleanData('val')
    cleanData('test')


data_name = "tweets"
split(data_name)








