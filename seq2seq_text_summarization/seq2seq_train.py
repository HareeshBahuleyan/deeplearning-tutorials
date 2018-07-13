import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import keras
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Optimizer
from keras import callbacks

import matplotlib as mpl
from matplotlib import pyplot as plt
from keras.utils import plot_model 
from IPython.display import Image

np.random.seed(17)


DATA_FILE_PATH = 'data/amazon_reviews.csv' 
EMB_DIR = '/collection/hpallika/deeplearning/research_work/word_embeddings/glove.6B.300d.txt'

MAX_TEXT_VOCAB_SIZE = 30000
MAX_SUMMARY_VOCAB_SIZE = 10000
MAX_TEXT_LEN = 100
MAX_SUMMARY_LEN = 5

LSTM_DIM = 300
EMBEDDING_DIM = 300

BATCH_SIZE = 128
N_EPOCHS = 100

print('[INFO] Reading data and pre-processing')
df = pd.read_csv(DATA_FILE_PATH)
print('Number of rows = ', df.shape[0])
df.head(10)

# Converting all columns to string
df['Summary'] = df['Summary'].apply(lambda x: str(x))
df['Text'] = df['Text'].apply(lambda x: str(x))


sent_len = lambda x:len(x)
df['Summary_length'] = df.Summary.apply(sent_len)
df[df['Summary_length']<5]['Summary'].tail()


# Summaries having lesser than 5 characters can be discarded - noisy data
indices = df[df['Summary_length']<5].index
df.drop(indices, inplace=True)

# Can drop the Summary_length columns - to save memory
df.drop('Summary_length', inplace=True, axis=1)

df.reset_index(inplace=True, drop=True)
df.shape

word_count = lambda x:len(x.split()) # Word count for each question
df['s_wc'] = df.Summary.apply(word_count)
df['t_wc'] = df.Text.apply(word_count)

p = 75.0

print(' Summary :{} % of the summaries have a length less than or equal to {}'.format(p, np.percentile(df['s_wc'], p)))
print(' Text :{} % of the texts have a length less than or equal to {}'.format(p, np.percentile(df['t_wc'], p)))

text_list = [' '.join(word_tokenize(x)[:MAX_TEXT_LEN]) for x in df['Text']]
text_list[:2]


summary_list = [' '.join(word_tokenize(x)[:MAX_SUMMARY_LEN]) for x in df['Summary']]
summary_list[:2]


filter_list = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

text_tokenizer = Tokenizer(filters=filter_list)
text_tokenizer.fit_on_texts(text_list)
print("Number of words in TEXT vocabulary:", len(text_tokenizer.word_index))

summary_tokenizer = Tokenizer(filters=filter_list)  
summary_tokenizer.fit_on_texts(summary_list)
print("Number of words in SUMMARY vocabulary:", len(summary_tokenizer.word_index))

text_word_index = {}
text_word_index['PAD'] = 0
text_word_index['UNK'] = 1
text_word_index['EOS'] = 2

for i, word in enumerate(dict(text_tokenizer.word_index).keys()):
    text_word_index[word] = i+3 # Move existing indices up by 3 places
    
text_tokenizer.word_index = text_word_index
X = text_tokenizer.texts_to_sequences(text_list)

# Replace OOV words with UNK token
# Append EOS to the end of all sentences
for i, seq in enumerate(X):
    if any(t>=MAX_TEXT_VOCAB_SIZE for t in seq):
        seq = [t if t<MAX_TEXT_VOCAB_SIZE else text_word_index['UNK'] for t in seq ]
    seq.append(text_word_index['EOS'])
    X[i] = seq    
    
# Padding and truncating sequences
X = pad_sequences(X, padding='post', truncating='post', maxlen=MAX_TEXT_LEN, value=text_word_index['PAD'])

# Finalize the dictionaries
text_word_index = {k: v for k, v in text_word_index.items() if v < MAX_TEXT_VOCAB_SIZE} 
text_idx_to_word = dict((i, word) for word, i in text_word_index.items()) 


summary_word_index = {}
summary_word_index['PAD'] = 0
summary_word_index['UNK'] = 1
summary_word_index['EOS'] = 2
summary_word_index['SOS'] = 3

for i, word in enumerate(dict(summary_tokenizer.word_index).keys()):
    summary_word_index[word] = i+4 # Move existing indices up by 4 places
    
summary_tokenizer.word_index = summary_word_index
Y = summary_tokenizer.texts_to_sequences(summary_list)

# Replace OOV words with UNK token
# Append EOS to the end of all sentences
for i, seq in enumerate(Y):
    if any(t>=MAX_SUMMARY_VOCAB_SIZE for t in seq):
        seq = [t if t<MAX_SUMMARY_VOCAB_SIZE else summary_word_index['UNK'] for t in seq ]
    seq.append(summary_word_index['EOS'])
    Y[i] = seq    
    
# Padding and truncating sequences
Y = pad_sequences(Y, padding='post', truncating='post', maxlen=MAX_SUMMARY_LEN, value=summary_word_index['PAD'])

# Finalize the dictionaries
summary_word_index = {k: v for k, v in summary_word_index.items() if v < MAX_SUMMARY_VOCAB_SIZE} 
summary_idx_to_word = dict((i, word) for word, i in summary_word_index.items()) 


print('[INFO] Prepare train-val-test splits')

X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.05)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

# Load GloVe word embeddings 
# Download Link: https://nlp.stanford.edu/projects/glove/
print("[INFO]: Reading Word Embeddings ...")
# Data path
embeddings = {}
f = open(EMB_DIR)
for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings[word] = vector
f.close()


encoder_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(text_word_index), EMBEDDING_DIM)) 

for word, i in text_word_index.items(): # i=0 is the embedding for the zero padding
    try:
        embeddings_vector = embeddings[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        encoder_embeddings_matrix[i] = embeddings_vector


print(encoder_embeddings_matrix.shape)



decoder_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(summary_word_index), EMBEDDING_DIM)) 

for word, i in summary_word_index.items(): # i=0 is the embedding for the zero padding
    try:
        embeddings_vector = embeddings[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        encoder_embeddings_matrix[i] = embeddings_vector
        
del embeddings


print(decoder_embeddings_matrix.shape)


from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

def batch_generator(X, Y, BATCH_SIZE):

    # this line is just to make the generator infinite, keras needs that    
    while True:
        
        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < len(X):
            
            [X_enc, X_dec], Y_dec = prepare_data(X, Y, batch_start, batch_end)
            
            yield ([X_enc, X_dec], Y_dec) 

            batch_start += BATCH_SIZE   
            batch_end += BATCH_SIZE


def prepare_data(X, Y, batch_start, batch_end):
    
    # Encoder input
    X_enc = X[batch_start:batch_end] 

    # Decoder input
    # Concatenate a column of 3s (i.e., SOS token) to the Y and remove the last element
    X_dec = np.c_[3 * np.ones([len(Y[batch_start:batch_end])]), Y[batch_start:batch_end, :-1]]

    # Decoder output - one hot encoded for softmax layer - 1 in |V|
    Y_dec = np.array([to_categorical(y, num_classes=len(summary_word_index)) for y in Y[batch_start:batch_end]])

    return [X_enc, X_dec], Y_dec


# Encoder Setup
enc_input = Input(shape=(MAX_TEXT_LEN, ), name='encoder_input')
enc_emb_look_up = Embedding(input_dim=MAX_TEXT_VOCAB_SIZE,
                             output_dim=EMBEDDING_DIM,
                             weights = [encoder_embeddings_matrix], 
                             trainable=False, 
                             mask_zero=True,
                             name='encoder_embedding_lookup')

enc_emb_text = enc_emb_look_up(enc_input)

encoder_lstm = LSTM(LSTM_DIM, return_state=True, name='encoder_lstm', dropout=0.2) # To return the final state of the encoder
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb_text)
encoder_states = [state_h, state_c] # Discard encoder_outputs (at each time step) and only keep the final states.


# Decoder Setup
dec_input = Input(shape=(None, ), name='decoder_input')
dec_emb_look_up = Embedding(input_dim=MAX_SUMMARY_VOCAB_SIZE,
                             output_dim=EMBEDDING_DIM,
                             weights = [decoder_embeddings_matrix], 
                             trainable=False, 
                             mask_zero=True,
                             name='decoder_embedding_lookup')

dec_emb_text = dec_emb_look_up(dec_input)

# We set up our decoder to return full output sequences,
# and to return internal LSTM states (h, c) as well. We don't use the 
# return states in the training model, but we will use them during inference.
decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True, name='decoder_lstm', dropout=0.2)

# Hidden state initialization using `encoder_states` as initial state.
decoder_outputs, _, _ = decoder_lstm(dec_emb_text,
                                     initial_state=encoder_states)

decoder_dense = Dense(MAX_SUMMARY_VOCAB_SIZE, activation='softmax', name='output_layer')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([enc_input, dec_input], decoder_outputs)
model.summary()

# Set optimizer and loss function 
optimizer = keras.optimizers.Adam(lr=0.001) # Try a different learning rate

loss = 'categorical_crossentropy'

filepath="saved_models/seq2seq_textsummarization_{epoch:02d}_{val_loss:.4f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath, 
                                       monitor='val_loss', 
                                       verbose=0, 
                                       save_best_only=False)
callbacks_list = [checkpoint]

model.compile(optimizer=optimizer, loss=loss)


STEPS_PER_EPOCH = len(X_train)//BATCH_SIZE
VAL_STEPS = len(X_val)//BATCH_SIZE


model.fit_generator(batch_generator(X_train, Y_train, BATCH_SIZE), 
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    epochs = N_EPOCHS,
                    validation_data = batch_generator(X_val, Y_val, BATCH_SIZE), 
                    validation_steps = VAL_STEPS, 
                    callbacks = callbacks_list,
                   )


