from torch import cuda
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import datetime

from model import wordEncoder, sentEncoder, HAN

def formatTime(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    :param elapsed: time in seconds.
    :return: time in hh:mm::ss format.
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def hanTrain(tweets1, sent_scores1, unique_tokens1):
    VOCAB_SIZE = len(unique_tokens1)
    NUM_CLASSES = len(set(sent_scores1))
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    HIDDEN_SIZE = 16
    EMBEDDING_DIM = 30
    DEVICE = 'cuda' if cuda.is_available() else 'cpu'
    # torch.device('cuda')

    word_encoder = wordEncoder(VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM).to(DEVICE)
    sent_encoder = sentEncoder(HIDDEN_SIZE * 2).to(DEVICE)
    model = HAN(word_encoder, sent_encoder, NUM_CLASSES, DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    loss = []
    weights = []
    # i = epoch
    #total_t0 = time.time()
    for epoch in tqdm(range(NUM_EPOCHS)):
        print("")
        print('============================== Epoch {:} / {:} =============================='.format(epoch + 1, NUM_EPOCHS))
        print('Training...')
        t0=time.time()
        current_loss = 0
        for step, tweet in enumerate(tweets1):
            sent, score = torch.tensor(tweet, dtype=torch.long).to(DEVICE), torch.tensor(sent_scores1[step]).to(DEVICE)
            word_weights, sent_weights, output = model(sent)
            #print("c_loss1: ", current_loss)
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = formatTime(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   Loss: {:>19,}   Elapsed: {:}.'.format(step, len(tweets1),
                                                                                           current_loss, elapsed))
            optimizer.zero_grad()
            current_loss += criterion(output.unsqueeze(0), score.unsqueeze(0))
            current_loss.backward(retain_graph=True)
            optimizer.step()
            #print("c_loss2: ", current_loss)

            loss.append(current_loss.item() / (step + 1))
    print(loss)
