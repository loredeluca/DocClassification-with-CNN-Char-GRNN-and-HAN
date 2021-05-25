from prep import preprocessing
from train import train

maxlen = 256 #indica frase di len max = 256 caratteri

(train_docs, train_labels), (test_docs, test_labels), \
    vocab, alphabet_size, check = preprocessing("./datasets/ag_news_csv", False)
print("\nStart Training+Test...\n")
train(train_docs, train_labels, test_docs, test_labels, vocab, alphabet_size, check, maxlen)
print("\n End Training.\n")

