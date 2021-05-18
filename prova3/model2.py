import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable



# Attenzione multipla con word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


# Modello wordRNN per generare un sentence vector
class WordRNN(nn.Module):
    def __init__(self, vocab_size,embedsize, batch_size, hid_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        ## Word Encoder
        self.embed = nn.Embedding(vocab_size, embedsize)
        self.wordRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Word Attention
        self.wordattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 2*hid_size,bias=False)
    def forward(self,inp, hid_state):
        emb_out  = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)

        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation),dim=1)

        sent = attention_mul(out_state,attn)
        return sent, hid_state


## The HAN model
class SentenceRNN(nn.Module):
    def __init__(self, vocab_size, embedsize, batch_size, hid_size, n_classes, max_seq_len):
        super(SentenceRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        self.cls = n_classes
        self.max_seq_len = max_seq_len
        self.wordRNN = WordRNN(vocab_size, embedsize, batch_size, hid_size)
        ## Sentence Encoder
        self.sentRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Sentence Attention
        self.sentattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)
        self.doc_linear = nn.Linear(2 * hid_size, n_classes)

    def forward(self, inp, hid_state_sent, hid_state_word):
        s = None
        ## Generating sentence vector through WordRNN
        for i in range(len(inp[0])):
            r = None
            for j in range(len(inp)):
                if (r is None):
                    r = [inp[j][i]]
                else:
                    r.append(inp[j][i])
            r1 = np.asarray([sub_list + [0] * (self.max_seq_len - len(sub_list)) for sub_list in r])
            _s, state_word = self.wordRNN(torch.cuda.LongTensor(r1).view(-1, self.batch_size), hid_state_word)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)

                out_state, hid_state = self.sentRNN(s, hid_state_sent)
        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        doc = attention_mul(out_state, attn)
        d = self.doc_linear(doc)
        cls = F.log_softmax(d.view(-1, self.cls), dim=1)
        print("CLASS: ", cls)
        return cls, hid_state

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self.batch_size, self.hid_size)).cuda()

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self.batch_size, self.hid_size)).cuda()

