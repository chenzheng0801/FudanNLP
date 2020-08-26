import numpy as np


def encode_seq(data_path):
    with open(data_path) as f:
        text = f.read()
    vocab = set(text)
    vocab_to_int = {w: c for c, w in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encode_text = [vocab_to_int[w] for w in text]
    return vocab_to_int, int_to_vocab, encode_text


def get_batches(arr, batch_size, n_steps, end_token):
    total_steps = batch_size * n_steps
    n_batches = len(arr) // total_steps
    arr = arr[: total_steps * n_batches]

    for i in range(0, len(arr), total_steps):
        x = np.array(arr[i: i+total_steps]).reshape((batch_size, n_steps))
        end_token_arr = np.array([end_token]*batch_size).reshape((batch_size, 1))
        y = np.concatenate((x[:, 1:], end_token_arr), axis=1)
        yield x, y