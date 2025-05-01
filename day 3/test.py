# train_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import pickle
import time

# Load preprocessed data
try:
    X = np.loadtxt('X.txt')  # Prompt indices (266x5)
    y = np.loadtxt('y.txt')  # First response word indices (266,)
except FileNotFoundError:
    print("Error: X.txt or y.txt not found. Run preprocess_jsonl.py first.")
    exit(1)
try:
    with open('vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
        word_to_idx = vocab_data['word_to_idx']
        idx_to_word = vocab_data['idx_to_word']
    vocab_size = len(word_to_idx)
except FileNotFoundError:
    print("Error: vocab.pkl not found. Run preprocess_jsonl.py first.")
    exit(1)

# Build simple LSTM model
model = Sequential([
    Embedding(vocab_size, 32, input_length=X.shape[1]),  # Convert indices to dense vectors
    LSTM(32, return_sequences=False),  # Simple LSTM for first word prediction
    Dense(vocab_size, activation='softmax')  # Predict word from vocab
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
start_time = time.time()
history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2, verbose=1)
print(f"Training took {time.time() - start_time:.2f} seconds")

# Save model
model.save('lstm_model.h5')
print("Model saved as lstm_model.h5")

# Test prediction
test_prompt = X[0:1]  # First prompt for demo
pred = model.predict(test_prompt, verbose=0)
pred_idx = np.argmax(pred, axis=1)[0]
pred_word = idx_to_word.get(pred_idx, '<UNK>')
print("Test prompt indices:", test_prompt)
print("Predicted response word:", pred_word)