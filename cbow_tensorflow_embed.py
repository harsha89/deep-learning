import numpy as np
import pandas as pd
from keras.utils import np_utils
from tensorflow.keras.preprocessing import sequence

training_sentences = []
training_labels = []

df_data = pd.read_csv('processed_all_data.csv', names=["link", "content", "label", "forum_type", "content_processed"])
df_data['content'] = df_data['content'].astype(str)
df_data['content_processed'] = df_data['content_processed'].astype(str)

df_african_american_general =  df_data[df_data["forum_type"] == "GENERAL_FORUM"]
print(df_african_american_general)


from keras.preprocessing import text

contents = df_african_american_general["content_processed"].to_numpy()
tokenizer = text.Tokenizer(lower=True)
tokenizer.fit_on_texts(contents)

word2id = tokenizer.word_index
word2id['PAD'] = 0

reverse_word_index = dict([(value, key) for (key, value) in word2id.items()])

# build vocabulary of unique words
id2word = {v:k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in contents]

embed_size = 100
window_size = 2 # context window size
vocab_size = len(word2id)

print('Vocabulary Size:', len(word2id))
print('Vocabulary Sample:', list(word2id.items())[:100])


def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size * 2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1

            context_words.append([words[i]
                                  for i in range(start, end)
                                  if 0 <= i < sentence_length
                                  and i != index])
            label_word.append(word)
            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            yield (x, y)


# Test this out for some samples
i = 0
for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])

        if i == 10:
            break
        i += 1


import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# build CBOW architecture
cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(GlobalAveragePooling1D())
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print(cbow.summary())

for epoch in range(1, 3):
    loss = 0.
    i = 0
    for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 100 == 0:
            print('Processed {} (context, word) pairs'.format(i))

        if i == 30000:
            break

    print('Epoch:', epoch, '\tLoss:', loss)
    print()


e = cbow.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

transpose_vectors = weights.T

indexes = list(range(1, 101))
words = []
i = 1
for word_num in range(0, vocab_size):
    word = reverse_word_index[word_num]
    words.append(word)


df = pd.DataFrame(data=transpose_vectors, index=indexes, columns=words)
df.to_csv("general_am.csv")

import io
import re

out_v = io.open('vecs_cbow_gen.tsv', 'w', encoding='utf-8')
out_m = io.open('meta_cbow_gen.tsv', 'w', encoding='utf-8')
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

    i = i + 1
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# embedding_df = pd.DataFrame(weights, index=list(id2word.values())[1:]).head()
# embedding_df.to_csv("ad_am.csv")


# from sklearn.metrics.pairwise import euclidean_distances
#
# # compute pairwise distance matrix
# distance_matrix = euclidean_distances(weights)
# print(distance_matrix.shape)
#
# # view contextually similar words
# similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1]
#                    for search_term in ['god', 'jesus', 'noah', 'egypt', 'john', 'gospel', 'moses','famine']}
#
# print(similar_words)