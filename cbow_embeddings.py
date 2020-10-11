import numpy as np
# import tensorflow_datasets as tfds
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
#
# train_data, test_data = imdb['train'], imdb['test']
#
# for s, l in train_data:
#     training_sentences.append(str(s.numpy()))
#     training_labels.append(l.numpy())
#
# for s, l in test_data:
#     testing_sentences.append(str(s.numpy()))
#     testing_labels.append(l.numpy())



df_gen_1 = pd.read_csv('forum_content.csv', names=["link", "content", "label"])
df_gen_2 = pd.read_csv('forum_content_gen.csv', names=["link", "content", "label"])
df_am_1 = pd.read_csv('forum_content_literature_am.csv', names=["link", "content", "label"])
df_am_2 = pd.read_csv('forum_content_am.csv', names=["link", "content", "label"])

all_df = pd.concat([df_gen_1,df_gen_2,df_am_1,df_am_2], ignore_index=True)

all_df = all_df.sample(frac = 1)
all_df.reset_index(inplace=True, drop=True)
all_df.dropna(inplace=True)
train, test  = train_test_split(all_df, test_size=0.2)

#str(s.tonumpy()) is needed in Python3 instead of just s.numpy()

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

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])
