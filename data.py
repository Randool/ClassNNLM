import codecs
from random import shuffle

import numpy as np
import torch

UNK = '<unk>'
PAD = '<pad>'
BOS = '<s>'
EOS = '</s>'


class Dictionary:
    def __init__(self, vocab_path):
        self.word2idx = {}
        self.idx2word = []
        self.word2cls = []
        self.cls_set = set()
        self.add_word(UNK)
        self.add_word(PAD)
        self.add_word(BOS)
        self.add_word(EOS)
        with codecs.open(vocab_path, 'r', 'utf8') as f:
            for line in f.read().splitlines():
                if len(line) == 0:
                    continue
                self.add_word(*line.split())
        self.ncls = len(self.cls_set)

    def add_word(self, word, cls=0):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word2cls.append(int(cls))
            self.cls_set.add(int(cls))
        return self.word2idx[word]

    def __getitem__(self, key):
        return self.word2idx.get(key, self.word2idx[UNK])

    def __len__(self):
        return len(self.idx2word)

    def __contains__(self, word):
        return word in self.word2idx

    def indices2sent(self, indices):
        return list(map(self.idx2word.__getitem__, indices))

    def sent2indices(self, sent):
        return list(map(self.__getitem__, sent))

    def get_class_chunks(self):
        # 返回连续的class片段
        def ascent_check(nums):
            for i in range(1, len(nums)):
                if nums[i] < nums[i - 1]:
                    return False
            return True

        assert (ascent_check(self.word2cls))
        cls_chunk_size = 1
        for i in range(1, len(self.word2cls)):
            if self.word2cls[i] != self.word2cls[i - 1]:
                yield cls_chunk_size
                cls_chunk_size = 0
            cls_chunk_size += 1
        yield cls_chunk_size


class DataIter:
    def __init__(self, corpus, batch_size, dictionary, cuda=False, training=False):
        self.batch_size = batch_size
        self.dictionary = dictionary
        self.cuda = cuda
        self.bos = dictionary[BOS]
        self.eos = dictionary[EOS]
        self.pad = dictionary[PAD]
        self.unk = dictionary[UNK]
        self.training = training

        self.lines = []
        reader = codecs.open(corpus, "r", "utf8") if isinstance(corpus, str) else corpus
        for line in reader:
            words = line.strip().split()
            self.lines.append([BOS] + words + [EOS])
        reader.close()

    def get_unigram_dist(self):
        dist = [0] * len(self.dictionary)
        for l in self.lines:
            # skip BOS
            for w in l[1:]:
                dist[self.dictionary[w]] += 1
        return torch.tensor(dist)

    def __iter__(self):

        def wrapper(d):
            return d.cuda() if self.cuda else d

        indices = list(range(len(self.lines)))
        if self.training:
            shuffle(indices)

        for batch in range(len(self)):
            batch_indices = indices[batch * self.batch_size: (batch + 1) * self.batch_size]
            batch_lines = [self.lines[idx] for idx in batch_indices]
            batch_lines.sort(key=lambda x: len(x), reverse=True)
            length = list(map(len, batch_lines))
            max_len = length[0]
            data = torch.zeros((len(batch_lines), max_len), dtype=torch.long).fill_(self.pad)
            for i, line in enumerate(batch_lines):
                data[i, :len(line)] = torch.tensor(self.dictionary.sent2indices(line), dtype=torch.long)
            data = wrapper(data)
            xs, ys = data[:, :-1], data[:, 1:]

            yield xs, ys, [_len - 1 for _len in length]

    def __len__(self):
        return len(self.lines) // self.batch_size


if __name__ == '__main__':
    import pdb

    data_path = './data/penn/'
    np.random.seed(1)

    dictionary = Dictionary(data_path + 'vocab.c.txt')
    batch_size = 20
    cuda = False
    train_iter = DataIter(
        corpus=data_path + 'valid.txt',
        batch_size=batch_size,
        dictionary=dictionary,
        cuda=cuda,
    )
    chunks = list(dictionary.get_class_chunks())
    pdb.set_trace()
    # for d in train_iter:
    #    break
