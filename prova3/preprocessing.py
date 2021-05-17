import pandas as pd
from tqdm import tqdm
import numpy as np
import re

def hanPreprocessing():
    data = pd.read_csv("datasets/twitter.csv", encoding='latin-1')
    print(data)
    print('end read', len(data))  # 7156 righe
    print(data.shape[0])

    NUM_INSTANCES = 3000  # lui ne prende 3000, ma io le prenderei tutte
    MAX_SENT_LEN = 10  # len(max(eng_sentences)), lui prende 10 ma per fare una cosa più precisa
    # dovrei prendere la lunghezza della frase a lunghezza massima
    tweets, sent_scores = [], []
    unique_tokens = set()

    for i in tqdm(range(NUM_INSTANCES)):
        rand_idx = np.random.randint(len(data))
        # find only letters in sentences
        tweet = []
        # per ogni text all'indice rand_idx, lo splitta quando c'è un punto
        # es: ['ciao sto bene. mi chiamo pino']-->['ciao sto bene', 'mi chiamo pino']
        sentences = data["text"].iloc[rand_idx].split(".")
        # per ogni sotto frase nella frase
        for sent in sentences:
            if len(sent) != 0:
                # divide ogni frase in parole
                # es: sent='ciao sono mario' --> sent='ciao','sono','mario'
                sent = [x.lower() for x in re.findall(r"\w+", sent)]
                # qui mette tutta la frase fino a max_sent_len se è più lunga
                # altrimenti inserisce <pad>
                if len(sent) >= MAX_SENT_LEN:
                    sent = sent[:MAX_SENT_LEN]
                else:
                    for _ in range(MAX_SENT_LEN - len(sent)):
                        sent.append("<pad>")
                # il risultato dell'elaborazione lo mette in tweet per la singola sent
                tweet.append(sent)
                # penso salvi le parole(tipo crea una specie di dizionario)
                unique_tokens.update(sent)
        # tutte le frasi le mette in tweets
        tweets.append(tweet)
        # qui contorlla la label, cioè come è valutata la frase
        if data["sentiment"].iloc[rand_idx] == 'not_relevant':
            sent_scores.append(0)
        else:
            sent_scores.append(int(data["sentiment"].iloc[rand_idx]))
    print("sentScore", sent_scores)
    unique_tokens = list(unique_tokens)
    print(unique_tokens)

    # print the size of the vocabulary
    print(len(unique_tokens))
    print(tweets)
    # per ogni frase in tweets(i), per ogni sottofrase nella frase(j)
    # encode each token into index
    for i in tqdm(range(len(tweets))):
        for j in range(len(tweets[i])):
            tweets[i][j] = [unique_tokens.index(x) for x in tweets[i][j]]

    #print(tweets)
    return tweets, sent_scores, unique_tokens
