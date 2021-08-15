# =====================================================================================================
# NLP problem
#
# In this exercise I try to build and train a classifier for the sarcasm dataset.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 80%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def nlp():
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.8 and logs.get('val_accuracy') > 0.8):
                print("\nReached desired accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    file = open("../dataset/sarcasm.json", 'r', encoding='utf-8')
    for line in file.readlines():
        row = json.loads(line)
        sentences.append(row['headline'])
        labels.append(row['is_sarcastic'])

    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.summary()

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model.fit(training_padded, training_labels, epochs=100,
              validation_data=(testing_padded, testing_labels),
              verbose=1, steps_per_epoch=20,
              validation_steps=5, callbacks=[callbacks])
    return model


# The code below is to save the model as a .h5 file.
if __name__ == '__main__':
    model = nlp()
    model.save("nlp.h5")
