import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils.dataloader import get_bbc_tokenized_bert

df_train_x, df_train_y, df_test_x, df_test_y = get_bbc_tokenized_bert(False, True, False)

max_length = 512
labels_num = 5

model = Sequential()
model.add(Embedding(100, 128, input_length=max_length))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(labels_num, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(df_train_x)

# Train the model
batch_size = 64
epochs = 5
model.fit(df_train_x.numpy(), df_train_y.numpy(), batch_size=batch_size, epochs=epochs)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(df_test_x.numpy(), df_test_y.numpy())
print(f'Test accuracy: {accuracy * 100:.2f}%')
