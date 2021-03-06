#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from CharRNN import CharRNN, make_initial_state
from utils import prepare_data

# input data
def load_data(args):
    vocab = {}
    print ('%s/input.txt'% args.data_dir)
    chars = open('%s/input.txt' % args.data_dir, 'rb').read().lower()
    words = list(chars.split(" "))
    chars = list(chars)
    
    for i, char in enumerate(chars):
        if char not in vocab:
            vocab[char] = len(vocab)
    vocab['GO'] = len(vocab)
    vocab_index_of_GO = vocab['GO']
    dataset = []
    window = 1
    
    for i, word in enumerate(words):
        for j in range(i-window, i+window):
            if j==i or j<0 or j>=len(words): continue
            entry_x=[]
 
            for char in word:
                entry_x.append(vocab[char])
              
            entry_x.append(vocab['GO'])
            
            for char in words[j]:
                entry_x.append(vocab[char])
                
            dataset.append(entry_x)
            
    #dataset = np.asarray(dataset, dtype=np.int32)
    #mask = np.asarray(mask, dtype=np.int32)
    print 'corpus word length:', len(words)
    print 'character vocab length:', len(vocab)
    return dataset, chars, vocab, vocab_index_of_GO

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',                   type=str,   default='data/test')
parser.add_argument('--checkpoint_dir',             type=str,   default='cv')
parser.add_argument('--gpu',                        type=int,   default=-1)
parser.add_argument('--rnn_size',                   type=int,   default=128)
parser.add_argument('--learning_rate',              type=float, default=2e-3)
parser.add_argument('--learning_rate_decay',        type=float, default=0.97)
parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)
parser.add_argument('--decay_rate',                 type=float, default=0.95)
parser.add_argument('--dropout',                    type=float, default=0.0)
parser.add_argument('--seq_length',                 type=int,   default=10)
parser.add_argument('--batchsize',                  type=int,   default=5)
parser.add_argument('--epochs',                     type=int,   default=50)
parser.add_argument('--grad_clip',                  type=int,   default=5)
parser.add_argument('--init_from',                  type=str,   default='')
parser.add_argument('--max_len',                    type=int,   default=10)
args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

n_epochs    = args.epochs
n_units     = args.rnn_size
batchsize   = args.batchsize
bprop_len   = args.seq_length
grad_clip   = args.grad_clip

train_data, chars, vocab, vocab_index_of_GO = load_data(args)
pickle.dump(vocab, open('%s/vocab.bin'%args.data_dir, 'wb'))

if len(args.init_from) > 0:
    model = pickle.load(open(args.init_from, 'rb'))
else:
    model = CharRNN(len(vocab), n_units)

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

optimizer = optimizers.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)
optimizer.setup(model.collect_parameters())

whole_len    = len(train_data)
n_batches    = whole_len / batchsize
train_data   = train_data[:n_batches*batchsize]
epoch        = 0
start_at     = time.time()
cur_at       = start_at
state        = make_initial_state(n_units, batchsize=batchsize)
if args.gpu >= 0:
    accum_loss   = Variable(cuda.zeros(()))
    for key, value in state.items():
        value.data = cuda.to_gpu(value.data)
else:
    accum_loss   = Variable(np.zeros((), dtype=np.float32))
print train_data
print 'going to train {} iterations'.format(n_batches * n_epochs)
for i in xrange(n_epochs):
  for j in xrange(n_batches):
    batch_data = train_data[j*batchsize:(j+1)*batchsize]
    batch_data, mask_data = prepare_data(batch_data, vocab_index_of_GO)
    #batch_data = batch_data.T
    #mask_data = mask[j*batchsize:(j+1)*batchsize]
    #mask_data = mask_data.T
    #assert batch_data.shape[0] == args.max_len
    #assert mask_data.shape[0] == args.max_len
    for timestep in xrange(len(batch_data)):
        x_batch = batch_data[timestep]
        y_batch = x_batch * mask_data[timestep]
        m_batch = mask_data[timestep].astype(np.float32) #this has to be converted here
        mask_bar = 1-m_batch
        m_batch = m_batch[:,None]
        
        #s_batch is the mask for probabilty distribution 
        s_batch = np.array(np.zeros((len(x_batch),len(vocab))), dtype=np.float32)
        s_batch[:,0] = mask_bar
        print x_batch
        if args.gpu >=0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)
            m_batch = cuda.to_gpu(m_batch)
            s_batch = cuda.to_gpu(s_batch)

        state, loss_i = model.forward_one_step(x_batch, y_batch, m_batch,s_batch, state, dropout_ratio=args.dropout)

        accum_loss   += loss_i
       
        if (timestep + 1) % bprop_len == 0:  # Run truncated BPTT
            now = time.time()
            print '{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)/bprop_len, n_batches,  accum_loss.data / bprop_len, now-cur_at)
            cur_at = now

            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            if args.gpu >= 0:
                accum_loss = Variable(cuda.zeros(()))
            else:
                accum_loss = Variable(np.zeros((), dtype=np.float32))

            optimizer.clip_grads(grad_clip)
            optimizer.update()
    
    if (i + 1) % 10000 == 0:
        fn = ('%s/charrnn_epoch_%.2f.chainermodel' % (args.checkpoint_dir, float(i)/n_batches))
        pickle.dump(copy.deepcopy(model).to_cpu(), open(fn, 'wb'))

    if (i + 1) % n_batches == 0:
        epoch += 1
        if epoch >= args.learning_rate_decay_after:
            optimizer.lr *= args.learning_rate_decay
            print 'decayed learning rate by a factor {} to {}'.format(args.learning_rate_decay, optimizer.lr)

    sys.stdout.flush()
