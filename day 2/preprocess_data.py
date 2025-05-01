import json
import numpy as np
from collections import Counter
import pickle

prompts = []
responses = []
line_numbers = 0
max_enteries = 266

try:
    with open('convo.jsonl','r',encoding='utf-8') as f:
        for line in f:
            line_numbers += 1
            if line_numbers > max_enteries:
                print(f"Stoping at max enteries {max_enteries}")
                break
            line = line.strip()
            if not line:
                print(f"Skipping empty line at {line_numbers}")
                continue
            try:
                data = json.loads(line)
                if 'prompt' not in data or 'response' not in data:
                    print(f"Skipping invalid file line number {line_numbers}")
                    continue
                prompts.append(data['prompt'].lower())
                responses.append(data['response'].lower().split()[0])
            except json.JSONDecodeError:
                print(f"Skipping invalid json at line {line_numbers}")
                continue
except FileNotFoundError:
    print("Error: convo.jsonl not found")
    sample_data = [
        { "prompt": "What’s a fun fact about Canada?", "response": "Canada has the longest coastline in the world—over 202,000 kilometers! That’s enough to circle Earth five times." },
        { "prompt": "What’s the significance of Mount Everest?", "response": "Mount Everest is the highest peak in the world, standing at 8,848 meters. Climbers from around the globe attempt to summit it, testing their limits against nature." },
        { "prompt": "What’s unique about Turkish culture?", "response": "Turkey bridges Europe and Asia, blending Eastern and Western influences. Turkish baths (hammams) and coffee are iconic cultural elements." },
        { "prompt": "What’s a fun fact about Antarctica?", "response": "Antarctica is the driest continent on Earth, despite being covered in ice. It’s also the only continent without a permanent human population." },
    ]    
    with open('convo.jsonl','w',encoding='utf-8') as f:
        for item in sample_data:
            json.dump(item,f)
            f.write('\n')
    prompts = [item['prompt'].lower() for item in sample_data]
    responses = [item['response'].lower().split()[0] for item in sample_data]
except UnicodeDecodeError as e:
    print(f"Error {e}")
    exit(1)

if not prompts or not responses:
    print("Error: No Valid data")
    exit(1)
    
max_len = 5
all_words = [word for prompt in prompts for word in prompt.split()]
word_count = Counter(all_words)
vocab = ['<PAD>','<UNK>'] + list(set(word_count.keys()))[:100]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

X = []
for prompt in prompts:
    words = prompt.split()[:max_len]
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
    indices += [word_to_idx['<PAD>']] * (max_len - len(indices))
    X.append(indices)
    
y = [word_to_idx.get(response, word_to_idx["<UNK>"]) for response in responses]

X = np.array(X)
y = np.array(y)


np.savetxt('X.txt',X)
np.savetxt('y.txt',y)
with open('vocab.pkl','wb') as f:
    pickle.dump({'word_to_idx': word_to_idx, 'idx_to_word':idx_to_word},f)
    
print("Preprocessed data: ",X.shape)
print("Sample Prompt: ", X[0])
print("sample response: ",y[0])
print("vocabs size: ", len(vocab))