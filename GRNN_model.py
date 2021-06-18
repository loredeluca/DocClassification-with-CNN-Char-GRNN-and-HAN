from enum import Enum
import torch
import torch.nn as nn


class GRNN(nn.Module):
    class SentenceModel(Enum):
        LSTM = 1
        CONV = 2

    def __init__(self, output_size, sentence_model, embedding_matrix, device):
        super(GRNN, self).__init__()

        self.sentence_model = sentence_model

        self.input_size = len(embedding_matrix[0])
        self.hidden_size = 50
        self.output_size = output_size

        #self.vocab_size = len(embedding_matrix)

        self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))

        self.word_linear = nn.Linear(self.input_size, self.hidden_size)

        if self.sentence_model == self.SentenceModel.CONV:
            self.conv1 = nn.Conv1d(self.hidden_size, self.hidden_size, 1, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, 2, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_size, self.hidden_size, 3, stride=1)
            self.conv = [self.conv1, self.conv2, self.conv3]
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1)

        self.tanh = nn.Tanh()

        # One GNN for the forward processing and one for the reverse direction
        self.gnn_f = GNN(self.hidden_size, self.hidden_size)
        self.gnn_b = GNN(self.hidden_size, self.hidden_size)

        self.linear = nn.Linear(2 * self.hidden_size, self.output_size)

        self.softmax = nn.Softmax()

        self.double()

        self.device = device
        self.cuda(self.device)

    def forward(self, doc):
        hidden_state = torch.zeros(1, self.hidden_size, requires_grad=True, dtype=torch.double, device=self.device)
        hidden_states = None
        sentence_reps = []
        num_sentences = len(doc)
        for i in range(0, num_sentences):
            # Turn vocabulary ids into embedding vectors
            sentence: torch.Tensor = self.word_embedding(torch.tensor(doc[i], dtype=torch.long, device=self.device))
            sentence = sentence.to(self.device)
            sentence.requires_grad = True
            num_words = len(sentence)

            if num_words == 0:
                continue

            sentence_new = None
            for word in sentence:
                out = self.word_linear(word)
                out = out.unsqueeze(0)
                sentence_new = out if sentence_new is None else torch.cat((sentence_new, out))

            sentence = sentence_new

            # Add third dimension for number of sentences (here: always equal to one)
            sentence = sentence.unsqueeze(2)

            # Model the sentences either with convolutional filters or with an LSTM
            if self.sentence_model == self.SentenceModel.CONV:
                sentence_rep = self.sentence_convolution(num_words, sentence)
            else:
                sentence_rep = self.sentence_lstm(sentence)

            sentence_reps.append(sentence_rep)

            # Model the document as GNN
            hidden_state = self.gnn_f(sentence_rep, hidden_state)

            # Concatenate GNN output (=hidden_state) of all sentences -> Tensor of dim [num_sentences, hidden_size]
            hidden_states = hidden_state if hidden_states is None else torch.cat((hidden_states, hidden_state))

        # Do backward processing too if required
        hidden_state_b = torch.zeros(1, self.hidden_size, requires_grad=True, dtype=torch.double, device=self.device)
        hidden_states_combined = None
        for i, sentence_rep in enumerate(reversed(sentence_reps)):
            hidden_state_b = self.gnn_b(sentence_rep, hidden_state_b)
            hidden_state_combined = torch.cat((hidden_states[-(i+1)].unsqueeze(0), hidden_state_b), dim=1)
            hidden_states_combined = hidden_state_combined if hidden_states_combined is None else torch.cat((hidden_state_combined, hidden_states_combined))

        gnn_out = hidden_states_combined.mean(0)

        # Finally, compute the output as softmax of a linear mapping
        output = self.softmax(self.linear(gnn_out))

        return output

    def sentence_convolution(self, num_words, sentence):
        # Rearrange shape for Conv1D layers
        sentence = sentence.permute(2, 1, 0)

        # We can't apply a convolution filter to an input that is smaller than the kernel size.
        # Hence, we apply one filter after the other with increasing kernel size until it exceeds input size.
        conv_result = None
        for kernel_size in range(1, 4):
            if num_words >= kernel_size:
                # Since the size of the sentences varies, we have to rebuild the avg pooling layer every iteration
                avg_pool_layer = nn.AvgPool1d(num_words - kernel_size + 1)
                avg_pool_layer.double()

                X = self.conv[kernel_size - 1](sentence)
                X = avg_pool_layer(X)
                X = self.tanh(X)

                # Concatenate results
                conv_result = X if conv_result is None else torch.cat((conv_result, X))
            else:
                break
        # In the end merge the output of all applied pooling layers by averaging them
        sentence_rep = conv_result.mean(0)
        return sentence_rep.t()

    def sentence_lstm(self, sentence):
        sentence = sentence.permute(0, 2, 1)
        initial_hidden_state = (torch.zeros(1, 1, self.hidden_size, dtype=torch.double, device=self.device),
                                torch.zeros(1, 1, self.hidden_size, dtype=torch.double, device=self.device))
        out, _ = self.lstm(sentence, initial_hidden_state)

        # LSTM output contains the whole state history for this sentence.
        # We only need the last output.
        sentence_rep = out[-1]
        return sentence_rep


class GNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(GNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(self.input_size * 2, self.hidden_size)
        self.forget_gate = nn.Linear(self.input_size * 2, self.hidden_size)
        self.input_gate = nn.Linear(self.input_size * 2, self.hidden_size)

    def forward(self, input, h):
        comp = torch.cat((input, h), dim=1)
        i_t = torch.sigmoid(self.input_layer(comp))
        f_t = torch.sigmoid(self.forget_gate(comp))
        g_t = torch.tanh(self.input_gate(comp))
        h_t = torch.tanh(i_t * g_t + f_t * h)  # "*" is element-wise multiplication here
        return h_t
