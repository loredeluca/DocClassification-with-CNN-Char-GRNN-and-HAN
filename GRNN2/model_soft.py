import torch
from torch import nn


class DocSenModel(nn.Module):
    def __init__(self, output_size, sentence_model, embedding_matrix, cuda=False):
        super(DocSenModel, self).__init__()

        self._sentence_model = sentence_model

        ###
        self._input_size = len(embedding_matrix[0])
        self._hidden_size = 50
        self._output_size = output_size

        self._vocab_size = len(embedding_matrix)
        ###

        self._word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))

        self._word_linear = nn.Linear(self._input_size, self._hidden_size)

        if self._sentence_model == 'conv':
            self._conv1 = nn.Conv1d(self._hidden_size, self._hidden_size, 1, stride=1)
            self._conv2 = nn.Conv1d(self._hidden_size, self._hidden_size, 2, stride=1)
            self._conv3 = nn.Conv1d(self._hidden_size, self._hidden_size, 3, stride=1)
            self._conv = [self._conv1, self._conv2, self._conv3]
        else:
            self._lstm = nn.LSTM(self._hidden_size, self._hidden_size, num_layers=1)

        self._tanh = nn.Tanh()

        # One GNN for the forward processing and one for the reverse direction
        self._gnn_f = GNN(self._hidden_size, self._hidden_size)

        self._gnn_b = GNN(self._hidden_size, self._hidden_size)
        self._linear = nn.Linear(2 * self._hidden_size, self._output_size)

        self._softmax = nn.Softmax()

        self.double()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''
        self._cuda = cuda
        if self._cuda:
            self._device = torch.device('cuda')
            self.cuda(self._device)
        else:
            self._device = torch.device('cpu')
        '''

    def forward(self, doc):
        num_sentences = len(doc)
        hidden_state = torch.zeros(1, self._hidden_size, requires_grad=True, dtype=torch.double, device=self._device)
        hidden_states = None
        sentence_reps = []
        for i in range(0, num_sentences):
            # Turn vocabulary ids into embedding vectors
            sentence: torch.Tensor = self._word_embedding(torch.tensor(doc[i], dtype=torch.long, device=self._device))
            sentence = sentence.to(self._device)
            sentence.requires_grad = True
            num_words = len(sentence)

            if num_words == 0:
                continue

            sentence_new = None
            for word in sentence:
                out = self._word_linear(word)
                out = out.unsqueeze(0)
                sentence_new = out if sentence_new is None else torch.cat((sentence_new, out))

            sentence = sentence_new

            # Add third dimension for number of sentences (here: always equal to one)
            sentence = sentence.unsqueeze(2)

            # Model the sentences either with convolutional filters or with an LSTM
            if self._sentence_model == 'conv':
                sentence_rep = self._sentence_convolution(num_words, sentence)
            else:
                sentence_rep = self._sentence_lstm(sentence)

            sentence_reps.append(sentence_rep)

            # Model the document as GNN
            hidden_state = self._gnn_f(sentence_rep, hidden_state)

            # Concatenate GNN output (=hidden_state) of all sentences -> Tensor of dim [num_sentences, hidden_size]
            hidden_states = hidden_state if hidden_states is None else torch.cat((hidden_states, hidden_state))

        # Do backward processing too if required

        hidden_state_b = torch.zeros(1, self._hidden_size, requires_grad=True, dtype=torch.double, device=self._device)
        hidden_states_combined = None
        for i, sentence_rep in enumerate(reversed(sentence_reps)):
            hidden_state_b = self._gnn_b(sentence_rep, hidden_state_b)
            hidden_state_combined = torch.cat((hidden_states[-(i+1)].unsqueeze(0), hidden_state_b), dim=1)
            hidden_states_combined = hidden_state_combined if hidden_states_combined is None else torch.cat((hidden_state_combined, hidden_states_combined))

        gnn_out = hidden_states_combined.mean(0)


        # Finally, compute the output as softmax of a linear mapping
        output = self._softmax(self._linear(gnn_out))

        return output

    def _sentence_convolution(self, num_words, sentence):
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

                X = self._conv[kernel_size - 1](sentence)
                X = avg_pool_layer(X)
                X = self._tanh(X)

                # Concatenate results
                conv_result = X if conv_result is None else torch.cat((conv_result, X))
            else:
                break
        # In the end merge the output of all applied pooling layers by averaging them
        sentence_rep = conv_result.mean(0)
        return sentence_rep.t()

    def _sentence_lstm(self, sentence):
        sentence = sentence.permute(0, 2, 1)
        # Todo: init hidden state randomly or with zeros?
        # initial_hidden_state = (torch.randn(1, 1, self._hidden_size, dtype=torch.double),
        #                         torch.randn(1, 1, self._hidden_size, dtype=torch.double))
        initial_hidden_state = (torch.zeros(1, 1, self._hidden_size, dtype=torch.double, device=self._device),
                                torch.zeros(1, 1, self._hidden_size, dtype=torch.double, device=self._device))
        out, _ = self._lstm(sentence, initial_hidden_state)

        # LSTM output contains the whole state history for this sentence.
        # We only need the last output.
        sentence_rep = out[-1]
        return sentence_rep


class GNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(GNN, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size

        self._input_layer = nn.Linear(self._input_size * 2, self._hidden_size)
        self._forget_gate = nn.Linear(self._input_size * 2, self._hidden_size)
        self._input_gate = nn.Linear(self._input_size * 2, self._hidden_size)

    def forward(self, input, h):
        """
        :param input: Sentence representation. Tensor [1, input_size]
        :param h: hidden state of the previous GNN cell. Tensor [1, hidden_size]
        :return: Output of the GNN cell (= new hidden state). Tensor [1, hidden_size]
        """
        comp = torch.cat((input, h), dim=1)
        i_t = torch.sigmoid(self._input_layer(comp))
        f_t = torch.sigmoid(self._forget_gate(comp))
        g_t = torch.tanh(self._input_gate(comp))
        h_t = torch.tanh(i_t * g_t + f_t * h)  # "*" is element-wise multiplication here
        return h_t
