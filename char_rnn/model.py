import utils
import config

import numpy as np

import keras
import keras.backend as K


class SentenceSamplerCallback(keras.callbacks.Callback):
    def __init__(self, start_phrase, char_to_idx, idx_to_char):
        self.start_phrase = start_phrase

        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

    def on_epoch_end(self, epoch, logs):
        strict = sample(self.model, self.start_phrase, self.char_to_idx,
                        self.idx_to_char, strict=True)
        not_strict = sample(self.model, self.start_phrase, self.char_to_idx,
                            self.idx_to_char, strict=False)

        print('epoch: %s' % epoch)
        print('strict: %s' % strict)
        print('not_strict: %s' % not_strict)
        print()


def get_next_output(model, string, char_to_idx):
    x = np.expand_dims(utils.encode(string, len(string), char_to_idx), axis=0)

    # Model is predicting on batch of 1.
    return np.squeeze(model.predict([x]), axis=0)


def sample(model, start, char_to_idx, idx_to_char, total_chars=512, strict=True):
    result = start

    for _ in range(total_chars - len(start)):
        proba = get_next_output(model, result, char_to_idx)[-1]

        if strict:
            char = utils.decode_single(proba, idx_to_char)
        else:
            idx = np.random.choice(range(len(char_to_idx)), p=proba)
            char = idx_to_char[idx]

        result += char

        if char == '<UNK>' or char == '<EOS>':
            break

    return result


def build_model(num_chars):
    input_tensor = keras.layers.Input(shape=(None, num_chars))

    net = input_tensor
    net = keras.layers.LSTM(256, return_sequences=True,
                            recurrent_activation='sigmoid')(net)
    net = keras.layers.LSTM(256, return_sequences=True,
                            recurrent_activation='sigmoid')(net)
    net = keras.layers.Dense(num_chars, activation='softmax')(net)

    model = keras.models.Model(inputs=[input_tensor], outputs=[net])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train(model, datagen, char_to_idx, idx_to_char):
    schedule_func = lambda epoch: 10 ** -(3 + epoch // 25)
    scheduler = keras.callbacks.LearningRateScheduler(schedule_func)

    sampler = SentenceSamplerCallback('Brady:', char_to_idx, idx_to_char)

    model.fit_generator(datagen, 1000,
                        epochs=100,
                        callbacks=[sampler, scheduler])


def main(filename, header=None):
    # Training data.
    # corpus = utils.load_csv(filename, header)
    corpus = utils.load_raw_text(filename)

    char_to_idx, idx_to_char = utils.get_mapping(corpus)

    # Yields training data.
    datagen = utils.generator(corpus, char_to_idx,
                              config.batch_size, config.seq_length)

    # Char-to-char RNN.
    model = build_model(len(char_to_idx))

    train(model, datagen, char_to_idx, idx_to_char)


if __name__ == '__main__':
    np.random.seed(0)

    main('input.txt')
    # main('Reviews_clean.csv', 'Text')
