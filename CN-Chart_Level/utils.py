import numpy as np


def encode_data(data, maxlen, vocab, alphabet_size, check):
    print("Encoding test data...\n")
    # Crea una matrice maxlen x alphabet_size (in questo caso 256(1014)x69)
    # e si crea una matrice 3D data x maxlen x alphabet_size
    # ogni carattere subisce la 'one-hot encode'
    # (caratteri non presenti nell'alphabet codificati con vettore tutto zero)

    input_data = np.zeros((len(data), maxlen, alphabet_size))
    for idx, sent in enumerate(data):
        # sent = ['ciao pippo']
        counter = 0
        sent_array = np.zeros((maxlen, alphabet_size))
        # separe lettera per lettera nella frase
        chars = list(sent.lower().replace(' ', ''))
        #  chars = ['c','i','a','o','p','i','p','p','o']
        # per ogni carattere della frase
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(alphabet_size, dtype=np.int)
                # verifica che il carattere sia nell'alphabet
                if c in check:
                    # ix = indice del carattere c
                    vocab_index = vocab[c]
                    # char_array : vettore (1xalphabet_size) di tutti 0 e
                    # 1 al posto del carattere
                    char_array[vocab_index] = 1
                # sent_array : matrice(maxlen x alphabet_size) ad ogni riga corrisponde
                # un vettore(=cioè il carattere), tutta la matrice corrisponde ad una frase
                # che al massimo ha max_len caratteri
                sent_array[counter, :] = char_array
                counter += 1
        input_data[idx, :, :] = sent_array  # matrice 3D, contenente tutte le frasi in modo 'one-hot encode'

    return input_data


def mini_batch_generator(x, y, vocab, alphabet_size, vocab_check, maxlen, batch_size=128):
    for i in range(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, alphabet_size, vocab_check)

        yield (input_data, y_sample)


def shuffle_matrix(x, y):
    stacked = np.hstack((np.array(x).T, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi
