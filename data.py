import itertools
import numpy as np


class SequenceDataGenerator:
    '''Generates sequences from a dataset.'''

    def __init__(self,
                 sequences,
                 phrase_length=64,
                 dim=64,
                 batch_size=512,
                 validation_percent=0.01,
                 is_validation=False):
        '''Initialize a SequenceDataGenerator.

        Arguments:

        sequences - The list of symbolic, integer sequences
            representing different "songs."

        phrase_length - The length of phrases to be generated
            batch_size - The number of phrases to be generated.

        dim - The dimension of the symbol space, i.e. the number of
            possible values of a sequence element.

        batch_size - The number of samples generated on each
            iteration.

        validation_pct - The fraction, between 0 and 1, of data to be
            used for validation.
            
        is_validation - True if we are drawing from the validation
            pool, False otherwise.
        '''

        self.sequences = sequences
        self.phrase_length = phrase_length
        self.dim = dim
        self.batch_size = batch_size

        # Reset the random seed, so that a call to the constructor
        # with is_validation=True followed by a call with
        # is_validation=False produces two complementary sets of
        # indices.
        np.random.seed(0)

        # Get the indices of all data points. A data point consists of
        # a sequence of length phrase_length followed by the label, or
        # next element in the sequence. Therefore, take all
        # subsequences of length phrase_length + 1.
        self.sequence_indices = idx_seq_of_length(self.sequences, phrase_length + 1)
        n_points = len(self.sequence_indices)
        if is_validation:
            self.data_indices = np.arange(n_points)[
                np.random.random(n_points) < validation_percent]
        else:
            self.data_indices = np.arange(n_points)[
                np.random.random(n_points) >= validation_percent]

        assert len(self.data_indices) > 0, 'No data selected for {}'.format(is_validation)


    def gen(self):
        '''Lazily generate an infinite stream of data batches.

        Each batch is a tuple with two entries: batch_size Xs and
        batch_size ys.
        '''

        while True:
            X_batch = np.zeros((self.batch_size, self.phrase_length, self.dim))
            y_batch = np.zeros((self.batch_size, self.dim))

            for batch_idx in range(self.batch_size):
                # Choose a random data point. Then extract the
                # sequence and label corresponding to it.
                seq_idx, phrase_start_idx = self.sequence_indices[
                    np.random.choice(self.data_indices)]
                X_batch[batch_idx,
                        range(self.phrase_length),
                        self.sequences[seq_idx][phrase_start_idx: phrase_start_idx + self.phrase_length]] = 1
                y_batch[batch_idx,
                        self.sequences[seq_idx][phrase_start_idx + self.phrase_length]] = 1

            yield (X_batch, y_batch)


def idx_seq_of_length(sequences, length):
    '''List the start indices of all sequences of the given length. Each
    start index is a pair of indices. The first specifies an element
    of sequences and the second specifies the index within that
    sequence.'''
    indices = []
    for i, seq in enumerate(sequences):
        if len(seq) >= length:
            indices.extend(itertools.product([i], range(len(seq) - length + 1)))
    return indices

