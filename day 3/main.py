import json
import numpy as np
from collections import Counter
import pickle

# Load and validate JSONL dataset
prompts = []
responses = []
line_number = 0
max_entries = 266  # Match dataset size
try:
    with open('data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            if line_number > max_entries:
                print(f"Stopping at {max_entries} entries.")
                break
            line = line.strip()
            if not line:
                print(f"Skipping empty line at line {line_number}")
                continue
            try:
                data = json.loads(line)
                if 'prompt' not in data or 'response' not in data:
                    print(f"Skipping invalid JSON at line {line_number}: missing prompt or response")
                    continue
                prompts.append(data['prompt'].lower())
                responses.append(data['response'].lower().split()[:10])  # Limit to 10 response words
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {line_number}: {line}")
                continue
except FileNotFoundError:
    print("Error: fixed_convo.jsonl not found. Creating sample dataset.")
    sample_data = [
        {"prompt": "Hi", "response": "Hello there!"},
        {"prompt": "What’s Python?", "response": "It’s a cool programming language!"},
        {"prompt": "I hate traveling", "response": "Sorry to hear that!"},
        {"prompt": "What’s the weather like?", "response": "It’s sunny out there!"},
        {"prompt": "What’s a fun fact about Greece?", "response": "Greece is the birthplace of democracy, philosophy, and the Olympic Games."},
        {"prompt": "What’s unique about Hawaiian culture?", "response": "Hawaii celebrates aloha spirit—a deep sense of kindness and hospitality."},
        {"prompt": "What’s a popular dish in Italy?", "response": "Pizza Margherita! Originating in Naples, it’s simple yet delicious."}
    ]
    with open('fixed_convo.jsonl', 'w', encoding='utf-8') as f:
        for item in sample_data:
            json.dump(item, f)
            f.write('\n')
    prompts = [item['prompt'].lower() for item in sample_data]
    responses = [item['response'].lower().split()[:10] for item in sample_data]
except UnicodeDecodeError as e:
    print(f"Error: Unicode decode issue in fixed_convo.jsonl: {e}. Use format_jsonl.py to fix.")
    exit(1)

# Validate dataset
if not prompts or not responses:
    print("Error: No valid data loaded. Check your JSONL file or use format_jsonl.py.")
    exit(1)

# Tokenize and build vocabulary
max_prompt_len = 5
max_response_len = 10
all_words = [word for prompt in prompts for word in prompt.split()] + \
            [word for response in responses for word in response]
word_counts = Counter(all_words)
vocab = ['<PAD>', '<UNK>'] + list(set(word_counts.keys()))[:150]  # Reduced vocab for speed
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Convert prompts to indices
X = []
for prompt in prompts:
    words = prompt.split()[:max_prompt_len]
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
    indices += [word_to_idx['<PAD>']] * (max_prompt_len - len(indices))
    X.append(indices)

# Convert responses to indices (first word for Episode 3)
y = [word_to_idx.get(response[0], word_to_idx['<UNK>']) for response in responses if response]

# Convert to numpy arrays
X = np.array(X)  # Shape: (266, 5)
y = np.array(y)  # Shape: (266,)

# Save data and vocabulary
np.savetxt('X.txt', X)
np.savetxt('y.txt', y)
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word}, f)

print("Preprocessed prompt shape:", X.shape)
print("Preprocessed response shape:", y.shape)
print("Sample prompt indices:", X[0])
print("Sample response index:", y[0])
print("Vocabulary size:", len(vocab))