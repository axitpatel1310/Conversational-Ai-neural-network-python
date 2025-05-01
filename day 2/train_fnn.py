import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import time

try:
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')
except FileNotFoundError:
    print("error files not found")
    exit(1)
try:
    with open('vocab.pkl','rb') as f:
        vocab_data = pickle.load(f)
        word_to_idx = vocab_data['word_to_idx']
        idx_to_word = vocab_data['idx_to_word']
    vocab_size = len(word_to_idx)
except FileNotFoundError:
    print("error vocab.pkl not found")
    exit(1)
    
model = Sequential([
    Dense(32,input_shape=(X.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(vocab_size,activation="softmax")
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

start_time = time.time()
history = model.fit(X,y,epochs=30,batch_size=8, validation_split=0.2, verbose=1)
print(f"Trainning Took {time.time() - start_time:.2f} seconds")

model.save('fnn_model.h5')
print("model saved")

test_prompt = X[0:1]
pred = model.predict(test_prompt,verbose=0)
pred_idx = np.argmax(pred, axis=1)[0]
pred_word = idx_to_word.get(pred_idx, '<UNK>')
print('Test Prompt Indices', test_prompt)
print('Predicted response',pred_word)