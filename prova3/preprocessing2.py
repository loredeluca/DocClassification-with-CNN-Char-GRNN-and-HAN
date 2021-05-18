import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from more_itertools import collapse
from itertools import chain
from gensim.models import Word2Vec
import torch
from nltk import word_tokenize



def clean_str(string, max_seq_len):
    """
    adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = BeautifulSoup(string, "lxml").text # trasforma codice html in testo
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s =string.strip().lower().split(" ")
    # imposta tutte le stringhe con stessa lunghezza
    if len(s) > max_seq_len:
        return s[0:max_seq_len]
    return s

## creates a 3D list of format paragraph[sentence[word]]
def create3DList(dataframe, col_text, max_sent_len, max_seq_len):
    text=[]
    # per ogni paragrafo ('docs') in text
    # es: DOCS: Awesome! Google driverless cars will help the blind travel more often; https://t.co/QWuXR0FrBpv
    for docs in dataframe[col_text].to_numpy(): #as_matrix():
        sents=[]
        n_words = 0
        # separa ogni paragrafo in frasi('sent') e ogni frase in parole
        # es: SEQ: Awesome - Google driverless... - ...
        # es: sents: [['awesome'], ['google', 'driverless', 'cars', 'will', 'help', 'the', 'blind', 'travel', 'more', 'often', 'https', 't'], ['co', 'qwuxr0frbpv']]
        for seq in "|||".join(re.split("[.?!]", docs)).split("|||"):
            sents.append(clean_str(seq,max_sent_len))
            # se frase è più lunga di max_seq_len tronca la frase
            # cioè se frase ha più di n_words
            if n_words >= (max_seq_len-1):
                break
            n_words= n_words+1
        # le frasi cosi splittate, le inserisco in text
        # text = dataframe['text'] ma con le parole splittate
        text.append(sents)
    return text


def wordFrequency(train, val, test):
    def wordFrequencyStep(texts, w_count):
        for texts in texts:
            for text in texts:
                for token in text:
                    w_count[token] += 1
        return w_count

    word_count = defaultdict(int)
    word_count = wordFrequencyStep(train, word_count)
    word_count = wordFrequencyStep(val, word_count)
    word_count = wordFrequencyStep(test, word_count)
    return word_count


def hanPreprocessing(data_name):
    data = pd.read_csv("datasets/"+data_name+".csv", encoding='latin-1')
    #data = pd.read_csv("datasets/imdb_reviews.csv")
    print(data.head())

    ## mark the columns which contains text for classification and target class
    col_text = 'text'
    col_target = 'sentiment' #'label'

    #find number of class
    #n_classes = len(np.unique(data[col_target]))
    cls_arr = np.sort(data[col_target].unique()).tolist()
    classes = len(cls_arr)
    print(cls_arr)
    #print(n_classes)

    #divide dataset in train,validation and test set (80,10,10)
    length = data.shape[0]
    train_len = int(0.8 * length)
    val_len = int(0.1 * length)

    train = data[:train_len]
    val = data[train_len:train_len + val_len]
    test = data[train_len + val_len:]

    # Fissa la lunghezza massima delle frasi in un paragrafo
    # e delle parole in una frase
    # TODO: importa metodo wordAndSentenceCounter() (commentato in utils di
    #  MachineLearning1 trova il numero medio di lunghezza di una frase)
    max_sent_len = 12
    max_seq_len = 25

    # divide la recensione in frasi e le frasi in parole creando una 3DList
    # NB: x_train,val,test contengono SOLO testo no label
    x_train = create3DList(train, col_text, max_sent_len, max_seq_len)
    x_val = create3DList(val, col_text, max_sent_len, max_seq_len)
    x_test = create3DList(test, col_text, max_sent_len, max_seq_len)
    print("x_train: {}".format(len(x_train)))
    print("x_val: {}".format(len(x_val)))
    print("x_test: {}".format(len(x_test)))

    # A QUESTO PUNTO x_train, x_val_ x test HANNO QUESTA FORMA:
    # [[['two', 'places', 'i', 'd', 'invest', 'all', 'my', 'money', 'if', 'i', 'could', '3d'], [''], ['']], [[],[]]]
    # EQUIVALE ALLA ROBA FATTA FINO A PRIMA DEL CALCOLO LUNGHEZZA DIZIONARIO e ENCODE IN preprocessing.py

    # elenco di stopwords + punteggiatura
    stoplist = stopwords.words('english') + list(string.punctuation)
    # stemmer mi permette di fare tokenization: running-->run
    # es: stemmer.stem("running") = 'run
    stemmer = SnowballStemmer('english')

    # per ogni paragrafo in x_train, per ogni frase di ogni paragrafo, per ogni parola di ogni frase fai lo stemmer,
    # se la parola non è una stopword o punteggiatura
    # equivale a:
    #for para in x_train:
    #    for sent in para:
    #        for word in sent:
    #            if word not in stoplist:
    #                stemmer.stem(word.lower())
    # al termine x_train,val, test _texts contengono frasi senza stoplist e tokenizzate

    x_train_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                     for para in x_train]
    x_test_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                    for para in x_test]
    x_val_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                   for para in x_val]

    # calcola frequenza delle parole
    # es: two': 39, 'place': 19, 'invest': 26, 'money': 24, 'could': 209
    word_count = wordFrequency(x_train_texts, x_val_texts, x_test_texts)

    # Rimuovo le parole con frequenza minore di 5 (cioè lascio quelle con frequenza>5)

    x_train_texts = [[[token for token in text if word_count[token] > 5] for text in texts] for texts in x_train_texts]
    x_test_texts = [[[token for token in text if word_count[token] > 5] for text in texts] for texts in x_test_texts]
    x_val_texts = [[[token for token in text if word_count[token] > 5] for text in texts] for texts in x_val_texts]

    # Faccio il merge di train,val e test_texts in un'unica lista per fare il word_embedding
    # --> forse aveva più senso splittare dopo
    # Sviluppiamo un word embeddings addestrando i nostri modelli word2vec su un corpus personalizzato
    # l'alternativa è usare un dataset pre-trained(Glove), in cui ad ogni parola è associata una stringa numerica
    texts = list(collapse(x_train_texts[:] + x_test_texts[:] + x_val_texts[:], levels=1))
    print("Start word2vec")
    word2vec = Word2Vec(texts, vector_size=200, min_count=5)
    word2vec.save("dictonary_"+dataname)

    # converte x_train,test,val_texts in una lista di indici, cioè sostituisce numeri al posto di parole
    x_train_vec = [[[word2vec.wv.key_to_index[token] for token in text] for text in texts] for texts in x_train_texts]
    x_test_vec = [[[word2vec.wv.key_to_index[token] for token in text] for text in texts] for texts in x_test_texts]
    x_val_vec = [[[word2vec.wv.key_to_index[token] for token in text] for text in texts] for texts in x_val_texts]

    weights = torch.FloatTensor(word2vec.wv.vectors)#.cuda()

    vocab_size = len(word2vec.wv)
    print(vocab_size)

    # Genero lista di label per ogni texts
    y_train = train[col_target].tolist()
    y_test = test[col_target].tolist()
    y_val = val[col_target].tolist()

    ## Converta la lista di lablel [1,2,1,3,..]
    # in tensori [tensor([1.]), tensor([2.]), tensor([1.]), tensor([3.]), ..]
    # sarebbe torch.FloatTensor([cls_arr.index(label)]).cuda()
    y_train_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_train]
    y_val_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_val]
    y_test_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_test]

    # Cerca la parola più lunga e la frase più lunga
    max_seq_len = max([len(seq) for seq in chain.from_iterable(x_train_vec + x_val_vec + x_test_vec)])
    max_sent_len = max([len(sent) for sent in (x_train_vec + x_val_vec + x_test_vec)])
    print("max_seq_len: ", max_seq_len, " & max_sent_len: ", max_sent_len)

    # calcolo il percentile:
    # I percentili esprimono la variabilità individuale rispetto alla popolazione generale.
    # es: il percentile che indica il numero di persone dello stesso sesso e popolazione in percentuale di statura:
    #     quindi se la statura di un individuo si colloca al 10° percentile, significa che il 10% della popolazione
    #     è più basso di lui, mentre il 90% è più alto. Il 50° percentile rappresenta la statura media.
    # NB: non so quale sia l'utilità, forse posso eliminare
    np.percentile(np.array([len(seq) for seq in chain.from_iterable(x_train_vec + x_val_vec + x_test_vec)]), 90)
    np.percentile(np.array([len(sent) for sent in (x_train_vec + x_val_vec + x_test_vec)]), 90)

    # riempio gli input, cioè Inserisco in x_train_vec tanti [0] fino ad arrivare al max_sent_len
    # es: se max_sent_len = 15
    #     [[188, 448, 316, 340, 26, 526], [1], [1], [1]]
    #     [[188, 448, 316, 340, 26, 526], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    X_train_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_train_vec]
    X_val_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_val_vec]
    X_test_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_test_vec]

    return X_train_pad, X_val_pad, X_test_pad, y_train_tensor, y_val_tensor, y_test_tensor, vocab_size, classes, weights, max_seq_len


#dataname = "twitter"
#hanPreprocessing(dataname)
