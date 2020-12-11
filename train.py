import argparse
import math
import sys
from os.path import join
import time

import torch

import model
from data import Dictionary, DataIter
from decoder import SMDecoder, ClassBasedSMDecoder, NCEDecoder


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/sms', help='location of the data corpus')
    parser.add_argument('--vocab', type=str, default='./data/sms/vocab2w', help='location of the vocabulary file')
    parser.add_argument('--emsize', type=int, default=128, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--bidir', action="store_true", help="<->")
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--decoder', type=str, default='sm', help="'sm' or 'cls' or 'nce'")
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')

    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--lr', type=float, default=1, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--nce_nsample', type=int, default=10)

    parser.add_argument('--save', type=str, default='params/model_2w_sm.pt', help='path to save the final model')
    parser.add_argument('--cont', action='store_true', help="Continue training")
    parser.add_argument('--cuda', type=int, default=-1, help='-1 for cpu 0,1,2.. for gpu')
    _args = parser.parse_args()

    print('{:=^30}'.format('all args'))
    for _arg in vars(_args):
        print(' '.join(map(str, (_arg, getattr(_args, _arg)))))
    print('{:=^30}'.format('all args'))

    return _args


###############################################################################
# Training code
###############################################################################

class Trainer(object):
    def __init__(self, model, args, train_iter, valid_iter, test_iter=None, max_epochs=50):
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.max_epochs = max_epochs
        self.args = args

    def __train(self, lr, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=.9)
        for batch, data in enumerate(self.train_iter):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try back propagating all the way to start of the dataset.
            optim.zero_grad()

            loss = self.model.loss(data)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            optim.step()

            total_loss += loss.data

            if batch % self.args.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / self.args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(self.train_iter), lr,
                    elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

            sys.stdout.flush()

    def train(self):
        # Loop over epochs.
        lr = self.args.lr
        best_val_loss = None

        if self.args.cont:
            with open(self.args.save, 'rb') as f:
                self.model = torch.load(f)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.max_epochs + 1):
                epoch_start_time = time.time()
                self.__train(lr, epoch)
                val_loss = self.evaluate(self.valid_iter)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, (time.time() - epoch_start_time), ))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(self.args.save, 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    with open(self.args.save, 'rb') as f:
                        self.model = torch.load(f)
                    lr /= 4.0

                    if lr < 0.01:
                        break

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(self.args.save, 'rb') as f:
            self.model = torch.load(f)

        if self.test_iter is not None:
            self.evaluate(self.test_iter, 'test')

    @torch.no_grad()
    def evaluate(self, data_source, prefix='valid'):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0
        for data in data_source:
            loss = self.model.loss(data)
            total_loss += loss.data
        ave_loss = total_loss.item() / len(data_source)
        print('| {0} loss {1:5.2f} | {0} ppl {2:8.2f}'.format(prefix, ave_loss, math.exp(ave_loss)))
        return ave_loss


if __name__ == '__main__':
    # Set the random seed manually for reproducibility.
    args = arg_parse()
    torch.manual_seed(args.seed)
    CUDA = False
    if torch.cuda.is_available():
        if args.cuda == -1:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda id")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.set_device(args.cuda)
            CUDA = True

    dictionary = Dictionary(args.vocab)
    print("Vocab size:", len(dictionary))

    eval_batch_size = 10

    train_iter = DataIter(join(args.data, 'train.txt'), args.batch_size, dictionary=dictionary, cuda=CUDA,
                          training=True)
    valid_iter = DataIter(join(args.data, 'valid.txt'), eval_batch_size, dictionary=dictionary, cuda=CUDA)
    test_iter = DataIter(join(args.data, 'test.txt'), eval_batch_size, dictionary=dictionary, cuda=CUDA)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntoken = len(dictionary)
    n_dir = 2 if args.bidir else 1

    if args.decoder == 'sm':
        decoder = SMDecoder(nhid=args.nhid * n_dir, ntoken=ntoken)
    elif args.decoder == 'cls':
        decoder = ClassBasedSMDecoder(
            nhid=args.nhid * n_dir,
            ncls=dictionary.ncls,
            word2cls=dictionary.word2cls,
            class_chunks=list(dictionary.get_class_chunks()),
        )
    elif args.decoder == 'nce':
        decoder = NCEDecoder(
            nhid=args.nhid * n_dir,
            ntoken=ntoken,
            noise_dist=train_iter.get_unigram_dist(),
            nsample=args.nce_nsample,
        )
    else:
        raise ValueError("decoder should be 'sm' or 'cls' or 'nce', got {}".format(args.decoder))

    model = model.RNNModel(
        ntoken=ntoken,
        ninp=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        decoder=decoder,
        dropout=args.dropout,
        bidir=args.bidir
    )

    if CUDA:
        model.cuda()

    trainer = Trainer(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        test_iter=test_iter,
        max_epochs=args.epochs,
        args=args
    )

    trainer.train()
