import cPickle
import gzip
import os
import numpy

def prepare_data(seqs, vocab_index_of_GO,  maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    axis flipped!
    """
    # x: a list of sequences
    lengths = [len(s) for s in seqs]
    # get rid of sequences that exceeds maxlen
    if maxlen is not None:
        new_seqs = []
        new_lengths = []
        for l, s in zip(lengths, seqs):
            if l < maxlen:
                new_seqs.append(s)
                new_lengths.append(l)
        lengths = new_lengths
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None
    # add the mask
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples),dtype=numpy.int32)
    x_mask = numpy.zeros((maxlen, n_samples),dtype=numpy.int32)
    for idx, s in enumerate(seqs):
        # the mask is only for the output
        x[:lengths[idx], idx] = s
        go_position = s.index(vocab_index_of_GO)
        x_mask[go_position:lengths[idx], idx] = 1.

    return x, x_mask
