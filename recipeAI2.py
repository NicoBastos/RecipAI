from __future__ import print_function
import pandas as pd
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import sys
import codecs
import json

from tensorflow import keras

SEQUENCE_LEN = 10
MIN_WORD_FREQUENCY = 4
STEP = 1
BATCH_SIZE = 32
maxlen = 10
step = 3
sentences = []
next_chars = []


def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index % len(sentence_list)]]] = 1
            index = index + 1
        yield x, y


def print_vocabulary(words_file_path, words_set):
    words_file = codecs.open(words_file_path, 'w', encoding='utf8')
    for w in words_set:
        if w != "\n":
            words_file.write(w+"\n")
        else:
            words_file.write(w)
    words_file.close()


def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
    keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                    beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                    moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                    beta_constraint=None, gamma_constraint=None)
    model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    seed_index = np.random.randint(len(sentences + sentences_test))
    seed = (sentences + sentences_test)[seed_index]
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        sentence = seed
        print('----- diversity:', diversity)
        print('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))

            for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            # changed to str()
            sentence = sentence[1:]
            sentence.append(next_word)
            sys.stdout.write(" " + next_word)
            sys.stdout.flush()
        print()

with open('recipesRaw.json', 'r') as f:
    data = json.load(f)
recipes = pd.DataFrame(data)
text = ''

for index, row in recipes.loc['instructions'].iteritems():
    text = text + str(row).lower()

text = text[:int(len(text) / 32)]
chars = sorted(list(set(text)))
print('total chars:', len(chars))

print('Corpus length in characters:', len(text))

text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
print('Corpus length in words:', len(text_in_words))

# Calculate word frequency
word_freq = {}
for word in text_in_words:
     word_freq[word] = word_freq.get(word, 0) + 1

ignored_words = set()
for k, v in word_freq.items():
    if word_freq[k] < MIN_WORD_FREQUENCY:
        ignored_words.add(k)

words = set(text_in_words)

print('Unique words before ignoring:', len(words))
print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
words = sorted(set(words) - ignored_words)
print('Unique words after ignoring:', len(words))


word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# cut the text in semi-redundant sequences of SEQUENCE_LEN words
sentences = []
next_words = []
ignored = 0
for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
    # Only add the sequences where no word is in ignored_words
    if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
        sentences.append(text_in_words[i: i + SEQUENCE_LEN])
        next_words.append(text_in_words[i + SEQUENCE_LEN])
    else:
        ignored = ignored + 1
print('Ignored sequences:', ignored)
print('Remaining sequences:', len(sentences))

# x, y, x_test, y_test
(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(
    sentences, next_words
)

model = get_model()

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='val_acc', patience=5)
callbacks_list = [print_callback, early_stopping]


history = model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                    steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                    epochs=75,
                    callbacks=callbacks_list,
                    validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                    validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)
model.save("recipAI.h5")

