# import tensorflow_datasets as tfds
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

df_data = pd.read_csv('word_embedding_final_train_debug.csv', names=["link", "content", "label", "forum_type", "content_processed"], skiprows=1)
print(df_data.shape)
df_data = df_data[df_data['label'] != "label"]
df_data = df_data[df_data['label'] != "is_am"]
print(df_data.shape)
df_data_test = pd.read_csv('word_embedding_final_test_debug.csv', names=["link", "content", "label", "forum_type", "content_processed"], skiprows=1)
df_data_test = df_data_test[df_data_test['label'] != "label"]
df_data_test = df_data_test[df_data_test['label'] != "is_am"]

df_data['label'] = df_data['label'].astype(int)
df_data_test['label'] = df_data_test['label'].astype(int)

df_data_cancer_forums = df_data_test[df_data_test["forum_type"] == "GENERAL_CANCER_FORUM"]
df_data_general_black_american_forums = df_data_test[df_data_test["forum_type"] == "AFRICAN_AMERICAN_GENERAL_FORUM"]
df_data_black_american_cancer_forums = df_data_test[df_data_test["forum_type"] == "AFRICAN_AMERICAN_CANCER_FORUM"]
df_data_test = df_data_test[df_data_test["forum_type"] == "GENERAL_FORUM"]

print("========================================")
print("GENERAL FORUM DATA SIZE")
print(df_data_test.shape)
print("GENERAL AMERICAN AFRICAN FORUM DATA SIZE")
print(df_data_general_black_american_forums.shape)
print("GENERAL BLACK AMERICAN CANCER FORUM DATA SIZE")
print(df_data_black_american_cancer_forums.shape)
print("GENERAL GENERAL CANCER FORUM DATA SIZE")
print(df_data_cancer_forums.shape)
print("========================================")
print(df_data)
print(df_data_test)

df_data['content'] = df_data['content'].astype(str)
df_data_test['content'] = df_data_test['content'].astype(str)

# def check_int(x):
#     print(x)
#     int(x)
#     return x
#
# df_data['label'] = df_data['label'].apply(check_int)
# df_data_test['label'] = df_data_test['label'].apply(check_int)

train, test = train_test_split(df_data, test_size=0.2)

unseen_test_set = []
unseen_test_set_labels = []

for row in df_data_test.iterrows():
    content = row[1]['content']
    content = content.replace("\\r\\n", " ")
    unseen_test_set.append(content)
    unseen_test_set_labels.append(row[1]['label'])

unseen_gen_cancer_forum = []
unseen_gen_cancer_forum_labels = []

for row in df_data_cancer_forums.iterrows():
    content = row[1]['content']
    content = content.replace("\\r\\n", " ")
    unseen_gen_cancer_forum.append(content)
    unseen_gen_cancer_forum_labels.append(row[1]['label'])

unseen_gen_aa_forum = []
unseen_gen_aa_forum_labels = []
for row in df_data_general_black_american_forums.iterrows():
    content = row[1]['content']
    content = content.replace("\\r\\n", " ")
    unseen_gen_aa_forum.append(content)
    unseen_gen_aa_forum_labels.append(row[1]['label'])

unseen_cancer_aa_forum = []
unseen_cancer_aa_forum_labels = []
for row in df_data_black_american_cancer_forums.iterrows():
    content = row[1]['content']
    content = content.replace("\\r\\n", " ")
    unseen_cancer_aa_forum.append(content)
    unseen_cancer_aa_forum_labels.append(row[1]['label'])
# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()

for row in train.iterrows():
    content = row[1]['content']
    content = content.replace("\\r\\n", " ")
    training_sentences.append(content)
    training_labels.append(row[1]['label'])

for row in test.iterrows():
    content = row[1]['content']
    content = content.replace("\\r\\n", " ")
    testing_sentences.append(content)
    testing_labels.append(row[1]['label'])

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
testing_unforseen_final = np.array(unseen_test_set_labels)

testing_gen_cancer_final = np.array(unseen_gen_cancer_forum_labels)
testing_gen_aa_final = np.array(unseen_gen_aa_forum_labels)
testing_cancer_aa_final = np.array(unseen_cancer_aa_forum_labels)

vocab_size = 20000
embedding_dim = 100
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding="post")
print(padded[:5])
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type, padding="post")

unseern_testing_sequences = tokenizer.texts_to_sequences(unseen_test_set)
unseen_padded = pad_sequences(unseern_testing_sequences, maxlen=max_length, truncating=trunc_type, padding="post")

#General cancer forums
unseern_gen_cancer_sequences = tokenizer.texts_to_sequences(unseen_gen_cancer_forum)
unseern_gen_cancer_padded = pad_sequences(unseern_gen_cancer_sequences, maxlen=max_length, truncating=trunc_type, padding="post")

#General american african forums
unseern_gen_aa_sequences = tokenizer.texts_to_sequences(unseen_gen_aa_forum)
unseern_gen_aa_padded = pad_sequences(unseern_gen_aa_sequences, maxlen=max_length, truncating=trunc_type, padding="post")

#General american african cancer forums
unseern_cancer_aa_sequences = tokenizer.texts_to_sequences(unseen_cancer_aa_forum)
unseern_cancer_aa_padded = pad_sequences(unseern_cancer_aa_sequences, maxlen=max_length, truncating=trunc_type, padding="post")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(padded[1]))
print(training_sentences[1])
print(sequences[1])

# Model Definition with LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

training = True

if training!=True:
    # LOading existing weight matrix
    new_df = pd.read_csv("weights_matrix.csv")
    print(new_df)
    print(new_df.shape)

    # model.load_weights("model_weights.h5")
    model = keras.models.load_model('model_weights_mod.h5')
    model.layers[0].set_weights([new_df])
    model.layers[0].trainable = False
    e = model.layers[0]
    weights = e.get_weights()[0]
else:
    num_epochs = 10
    history = model.fit(padded, training_labels_final, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels_final), verbose=2)
    model.save_weights("model_weights.h5")
    model.save("model_weights_mod.h5")
    e = model.layers[0]
    weights = e.get_weights()[0]
    df = pd.DataFrame(data=weights)
    print(df.shape)
    print(df)
    df.to_csv("weights_matrix.csv", index=False)
    print(weights.shape)  # shape: (vocab_size, embedding_dim)

loss, accuracy = model.evaluate(unseen_padded, testing_unforseen_final, verbose=2)
print("==============================")
print('Accuracy for general forums: %f' % (accuracy*100))
print("==============================")

loss, accuracy = model.evaluate(unseern_gen_cancer_padded, testing_gen_cancer_final, verbose=2)
print("==============================")
print('Accuracy for general cancer forums: %f' % (accuracy*100))
print("==============================")

loss, accuracy = model.evaluate(unseern_gen_aa_padded, testing_gen_aa_final, verbose=2)
print("==============================")
print('Accuracy for african american general forums: %f' % (accuracy*100))
print("==============================")

loss, accuracy = model.evaluate(unseern_cancer_aa_padded, testing_cancer_aa_final, verbose=2)
print("==============================")
print('Accuracy for african american cancer forums: %f' % (accuracy*100))
print("==============================")

import io
import re

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
s_pre = "$"
i = 0
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    if "\n" in word:
        word = word.replace("\n", " ")

    word = word.strip()
    word = re.sub("[^0-9a-zA-Z]+", "1", word)

    if word == "":
        word = s_pre
        s_pre = s_pre + "$"

    # print(word)
    # print(i)
    i = i + 1
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# num_epochs = 50
# history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
#
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
