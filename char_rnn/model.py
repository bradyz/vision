import utils
import config

import numpy as np

import keras
import keras.backend as K


def loss_function(mask_tensor, eps=1e-8):
    def loss(y_true, y_pred):
        return -K.mean(y_true * K.log(y_pred + eps)) * mask_tensor

    return loss


def build_model(input_shape):
    input_tensor = keras.layers.Input(shape=input_shape)
    mask_tensor = keras.layers.Input(shape=[input_shape[0]])

    net = input_tensor
    net = keras.layers.LSTM(256, return_sequences=True)(net)
    net = keras.layers.Dense(input_shape[1], activation='softmax')(net)

    model = keras.models.Model(inputs=[input_tensor, mask_tensor],
                               outputs=[net])

    model.compile(optimizer='adam',
                  loss=loss_function(mask_tensor),
                  metrics=['accuracy'])

    return model


def get_next_output(model, string, char_to_idx):
    x = np.zeros((1, config.seq_length, len(char_to_idx)))

    for i, char in enumerate(string):
        x[0,i,char_to_idx[char]] = 1.0

    mask = np.ones((1, config.seq_length))

    # Model is predicting on batch of 1.
    return model.predict([x, mask])[0]


def sample(model, start, char_to_idx, idx_to_char, sample_size=32):
    result = start

    for i in range(sample_size):
        output = utils.decode_output(
                get_next_output(model, result, char_to_idx),
                idx_to_char)

        result += utils.decode_single(output[i], idx_to_char)

    return result


def train(model, datagen, char_to_idx, idx_to_char):
    for iteration, x_y_mask in enumerate(datagen):
        x, y, mask = x_y_mask

        model.train_on_batch([x, mask], [y])

        if iteration % 25 == 0:
            sequences = model.predict([x, mask])

            true = utils.decode_output(x[0], idx_to_char)
            pred = utils.decode_output(sequences[0], idx_to_char)

            print(true)
            print(pred)
            print()

            char = utils.decode_single(x[0][0], idx_to_char)

            tmp = sample(model, char, char_to_idx, idx_to_char)

            print(tmp)


def main(csvfile, header):
    # Training data.
    corpus = utils.load_csv(csvfile, header)
    char_to_idx, idx_to_char = utils.get_mapping(corpus)

    # Yields training data.
    datagen = utils.generator(corpus, char_to_idx,
                              config.batch_size, config.seq_length)

    # Char-to-char RNN.
    model = build_model((config.seq_length, len(char_to_idx)))

    train(model, datagen, char_to_idx, idx_to_char)


if __name__ == '__main__':
    np.random.seed(0)

    main('Reviews.csv', 'Text')
