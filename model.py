#!/usr/bin/env python
# encoding: utf-8
import torch.nn as nn

from utils import length2mask


class RNNModel(nn.Module):
    """ Container module with an encoder, a recurrent module, and a decoder. """

    def __init__(self, ntoken, ninp, nhid, nlayers, decoder, dropout=0.5, bidir=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True, bidirectional=bidir)
        self.drop = nn.Dropout(dropout)
        self.decoder = decoder

        self.init_weights()

        self.nhid = 2 * nhid if bidir else nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.init_weights()

    def forward(self, inputs, length=None):
        emb = self.encoder(inputs)
        output, _ = self.rnn(emb)
        output = self.drop(output)
        return output

    def loss(self, data):
        # forward rnn
        inputs, target, length = data   # [batch, seq_len], [batch, seq_len], [batch]
        rnn_output = self(inputs, length)   # [batch, seq_len, dim]

        # discard the pad
        mask = length2mask(length)  # shape: [batch, seq_len]
        # mask --> [batch, seq_len, 1] --> [batch, seq_len, dim]
        rnn_output = rnn_output.masked_select(mask.unsqueeze(dim=2).expand_as(rnn_output)).view(-1, self.nhid)
        target = target.masked_select(mask)

        # forward decoder and calculate loss
        # rnn_output: [xx, dim]; target: [xx]
        decoder_loss = self.decoder.forward_with_loss(rnn_output, target)

        return decoder_loss

    def forward_all(self, data, length):
        output = self(data, length)
        return self.decoder.forward_all(output, length)
