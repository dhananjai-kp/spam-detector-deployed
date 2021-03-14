# import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,Input,LSTM,Dense,Bidirectional,Dropout, Activation
from keras.models import Model
from tensorflow.keras.models import Sequential
tf.__version__
import warnings
warnings.filterwarnings("ignore")
import pickle

df = pd.read_csv('../input/spam-filter/emails.csv')

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

df.text = df.text.apply(process_text)

vocab_size = 10000
max_len = 250
tok = Tokenizer(num_words=vocab_size)
tok.fit_on_texts(df.text)
sequences = tok.texts_to_sequences(df.text)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

pickle.dump(tok, open('token.pkl', 'wb'))

sequences_matrix[0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sequences_matrix, df.spam, test_size = 0.2, random_state = 1)

model = Sequential()
model.add(Embedding(vocab_size, 200, input_length=max_len))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=64)

model.save('./model.h5')

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'lstm-model.pkl'
pickle.dump(model, open(filename, 'wb'))

# from keras.models import load_model
# model1 = load_model('./model.h5')

# scores = model1.evaluate(X_test, y_test, verbose=0)
# y_pred = model1.predict_classes(X_test)

# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
# print('confusion matrix:\n', confusion_matrix(y_pred,y_test))