import argparse
import numpy as np
import re

import pandas as pd
from keras.models import load_model
import json
sentences = []
next_words = []
with open('recipesRaw.json', 'r') as f:
    data = json.load(f)
recipes = pd.DataFrame(data)
text = ''

for index, row in recipes.loc['instructions'].iteritems():
    text = text + str(row).lower()

text = text[:int(len(text) / 16)]

text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
ignored_words = set()
word_freq = {}
for k, v in word_freq.items():
    if word_freq[k] < 4:
        ignored_words.add(k)

vocabulary = set(text_in_words)

vocabulary = [re.sub(r'(\S+)\s+', r'\1', w) for w in vocabulary]
vocabulary = sorted(set(vocabulary))

word_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_word = dict((i, c) for i, c in enumerate(vocabulary))

for i in range(0, len(text_in_words) - 15, 1):
    # Only add the sequences where no word is in ignored_words
    if len(set(text_in_words[i: i+15+1]).intersection(ignored_words)) == 0:
        sentences.append(text_in_words[i: i + 15])
        next_words.append(text_in_words[i + 15])
    else:
        ignored = ignored + 1


seed_index = np.random.randint(len(sentences))
seed = (sentences)[seed_index]

from keras.models import load_model
model = load_model('recipAI.h5')


def validate_seed(vocabulary, seed):
    """Validate that all the words in the seed are part of the vocabulary"""
    print("\nValidating that all the words in the seed are part of the vocabulary: ")
    seed_words = seed.split(" ")
    valid = True
    for w in seed_words:
        print(w, end="")
        if w in vocabulary:
            print(" ✓ in vocabulary")
        else:
            print(" ✗ NOT in vocabulary")
            valid = False
    return valid


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, indices_word, word_indices, seed,
                  sequence_length, diversity, quantity):
    """
    Similar to lstm_train::on_epoch_end
    Used to generate text using a trained model
    :param model: the trained Keras model (with model.load)
    :param indices_word: a dictionary pointing to the words
    :param seed: a string to be used as seed (already validated and padded)
    :param sequence_length: how many words are given to the model to generate
    :param diversity: is the "temperature" of the sample function (usually between 0.1 and 2)
    :param quantity: quantity of words to generate
    :return: Nothing, for now only writes the text to console
    """
    beforeSplitStencence = ""
    for i in seed:
        beforeSplitStencence = beforeSplitStencence + i
        # beforeSplitStencence = beforeSplitStencence + ' '
    sentence = beforeSplitStencence.split(" ")
    print("----- Generating text")
    print('----- Diversity:' + str(diversity))
    print('----- Generating with seed:\n"' + beforeSplitStencence)

    print(seed)
    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length, len(vocabulary)))
        for t, word in enumerate(sentence):
            x_pred[0, t, word_indices[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)

        print(" "+next_word, end="")
    print("\n")
generate_text(model, indices_word, word_indices, seed , len(seed), 0.15,20)
