import csv

import numpy as np


ASCII = 256
EOS = ASCII + 1
UNK = ASCII + 2
MAX_VAL = ASCII + 3


def load_csv(filename, header):
    data = list()

    with open(filename, 'r') as csvfile:
        debug = 1

        for row in csv.DictReader(csvfile):
            data.append(row[header])

            debug += 1
            if debug == 10:
                break

    return data


def encode(sequence, seq_length):
    x = np.zeros((seq_length, MAX_VAL), dtype=np.float32)

    for i, char in enumerate(sequence):
        if ord(char) < ASCII:
            x[i,ord(char)] = 1.0
        else:
            x[i,UNK] = 1.0

    if len(sequence) < seq_length:
        x[len(sequence),EOS] = 1.0

    x[len(sequence)+1:,UNK] = 1.0

    return x


def get_mask(sequence, seq_length):
    mask = np.zeros((seq_length), dtype=np.float32)
    mask[:len(sequence)] = 1.0
    mask[len(sequence):] = 0.0

    return mask


def decode(sequence):
    result = list()

    for i in range(sequence.shape[0]):
        for j in range(sequence.shape[1]):
            if sequence[i,j] == 1.0:
                if j < ASCII:
                    result.append(chr(j))
                elif j == EOS:
                    result.append('<EOS>')
                elif j == UNK:
                    result.append('<UNK>')

    return ''.join(result)


def generator(data, batch_size=16, seq_length=32):
    n = len(data)

    x = np.zeros((batch_size, seq_length, MAX_VAL), dtype=np.float32)
    y = np.zeros((batch_size, seq_length, MAX_VAL), dtype=np.float32)
    mask = np.zeros((batch_size, seq_length), dtype=np.float32)

    while True:
        for i in range(batch_size):
            index = np.random.randint(n)
            m = len(data[index])

            j = np.random.randint(m)

            x[i] = encode(data[index][j:j+seq_length], seq_length)
            y[i] = encode(data[index][j+1:j+seq_length+1], seq_length)
            mask[i] = get_mask(data[index][j:j+seq_length], seq_length)

        yield x, y, mask
