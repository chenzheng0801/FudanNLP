import numpy as np


class TextConverter:
    def __init__(self, text):
        vocab = set(text)
        self.vocab = vocab
        self.vocab_to_int = {w: c for c, w in enumerate(vocab)}
        self.int_to_vocab = dict(enumerate(vocab))

    def encode_seq(self, text):
        encode_indices = [self.vocab_to_int[w] for w in text]
        return encode_indices

    def decode_seq(self, indices):
        decode_text = [self.int_to_vocab[idx] for idx in indices]
        return decode_text

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.vocab:
            return self.vocab_to_int[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_vocab[index]
        else:
            raise Exception('Unknown index!')


def get_batches(arr, batch_size, n_steps, end_token):

    def generate_batch():
        nonlocal arr
        total_steps = batch_size * n_steps
        n_batches = len(arr) // total_steps
        arr = arr[: total_steps * n_batches]

        for i in range(0, len(arr), total_steps):
            x = np.array(arr[i: i+total_steps]).reshape((batch_size, n_steps))
            end_token_arr = np.array([end_token]*batch_size).reshape((batch_size, 1))
            y = np.concatenate((x[:, 1:], end_token_arr), axis=1)
            yield x, y

    return generate_batch
