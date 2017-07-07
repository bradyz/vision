import utils
import config

import numpy as np

import keras
import keras.backend as K


def loss_function(mask_tensor, eps=1e-8):
    def loss(y_true, y_pred):
        if y_pred.shape.ndims == 2:
            return K.constant(0.0)

        return -K.mean(y_true * K.log(y_pred + eps)) * mask_tensor

    return loss


def build_model(input_shape):
    input_tensor = keras.layers.Input(shape=input_shape)
    mask_tensor = keras.layers.Input(shape=[input_shape[0]])

    net = input_tensor
    net = keras.layers.LSTM(256, return_sequences=True, return_state=True)(net)

    sequences, _, state = net

    pred = keras.layers.Dense(input_shape[1], activation='softmax')(sequences)

    model = keras.models.Model(inputs=[input_tensor, mask_tensor],
                               outputs=[pred, state])

    model.compile(optimizer='adam',
                  loss=loss_function(mask_tensor),
                  metrics=['accuracy'])

    return model


def get_next_output(model, char, char_to_idx, state=None):
    x = np.zeros((1, 1, len(char_to_idx)))
    x[0,0,char_to_idx[char]] = 1.0

    # Make x the shape of (1, 1, chars).
    x = K.constant(x)
    mask = K.constant(np.ones(1))

    sequences, state = model([x, mask], state)

    sequences = sequences.eval(session=K.get_session())
    state = state.eval(session=K.get_session())

    return sequences, state


def keras_call(model, sequence, char_to_idx, idx_to_char):
    print(utils.decode_output(sequence, idx_to_char))

    result = [utils.decode_single(sequence[0], idx_to_char)]

    char = result[0]
    state = None

    for _ in range(sequence.shape[0]):
        import pdb; pdb.set_trace()
        new_char, new_state = get_next_output(model, char, char_to_idx, state)
        import pdb; pdb.set_trace()

        result.append(new_char)

        char = new_char
        state = new_state

    print(''.join(result))


def train(model, datagen, char_to_idx, idx_to_char):
    # Necessary since the loss has two outputs.
    hack = np.zeros((config.batch_size, len(char_to_idx)))

    for iteration, x_y_mask in enumerate(datagen):
        x, y, mask = x_y_mask

        model.train_on_batch([x, mask], [y, hack])

        if iteration % 5 == 0:
            sequences, state = model.predict([x, mask])

            sequences = sequences[0]
            state = state[0]

            true = utils.decode_output(x[0], idx_to_char)
            pred = utils.decode_output(sequences, idx_to_char)

            print(true)
            print(pred)
            print()

            keras_call(model, sequences, char_to_idx, idx_to_char)

            import pdb; pdb.set_trace()


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
    main('Reviews.csv', 'Text')
