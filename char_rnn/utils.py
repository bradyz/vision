import csv

import numpy as np


def load_csv(filename, header, num_samples=None):
    data = list()

    with open(filename, 'r') as csvfile:
        for i, row in enumerate(csv.DictReader(csvfile)):
            if num_samples and i >= num_samples:
                break

            text = ' '.join(row[header].split())

            if not text:
                continue

            data.append(text)

    return data


def load_raw_text(filename, line_count=50):
    corpus = list()
    sample = ''

    for i, row in enumerate(open(filename, 'r').read().split('\n')):
        sample += row + '\n'

        if i > 0 and i % line_count == 0:
            corpus.append(sample)
            sample = ''

    return corpus


def encode(sequence, seq_length, char_to_idx):
    x = np.zeros((seq_length, len(char_to_idx)), dtype=np.float32)

    for i, char in enumerate(sequence):
        x[i,char_to_idx[char]] = 1.0

    if len(sequence) < seq_length:
        x[len(sequence),char_to_idx['<EOS>']] = 1.0

    return x


def decode(sequence, idx_to_char):
    result = list()

    for i in range(sequence.shape[0]):
        for j in range(sequence.shape[1]):
            if sequence[i,j] == 1.0:
                result.append(idx_to_char[j])

    return ''.join(result)


def generator(data, char_to_idx, batch_size, seq_length):
    max_val = len(char_to_idx)

    x = np.zeros((batch_size, seq_length, max_val), dtype=np.float32)
    y = np.zeros((batch_size, seq_length, max_val), dtype=np.float32)

    while True:
        for i in range(batch_size):
            index = np.random.randint(len(data))
            j = np.random.randint(len(data[index]))

            x[i] = encode(data[index][j:j+seq_length],
                          seq_length, char_to_idx)
            y[i] = encode(data[index][j+1:j+seq_length+1],
                          seq_length, char_to_idx)

        yield x, y


def get_mapping(data):
    characters = set()

    for sequence in data:
        for char in sequence:
            characters.add(char)

    char_to_idx = {char: i for i, char in enumerate(sorted(characters))}
    idx_to_char = {i: char for i, char in enumerate(sorted(characters))}

    idx_to_char[len(idx_to_char)] = '<EOS>'
    char_to_idx['<EOS>'] = len(char_to_idx)

    return char_to_idx, idx_to_char


def decode_output(y, idx_to_char):
    return ''.join(idx_to_char.get(idx, '<UNK>') for idx in np.argmax(y, axis=1))


def decode_single(y, idx_to_char):
    return idx_to_char.get(np.argmax(y), '<UNK>')
