import json 
import pandas as pd
from collections import Counter

prompts = []
responses = []

try:
    with open('sample.jsonl','r') as f:
        for line in f:
            try: 
                data = json.loads(line.strip())
                prompts.append(data['prompt'])
                responses.append(data['response'])
            except json.JSONDecodeError:
                print("Skipping invalid json")
                continue
            
except FileNotFoundError:
    print("Error Sample json not found... lets create one")
    
    sample_data = [
        {
      "prompt": "What will education be like in the future?",
      "response": "Education might become hyper-personalized, using AI tutors and immersive VR classrooms. Students could learn at their own pace while exploring subjects through hands-on simulations."
    },
    {
      "prompt": "Will we ever live on other planets?",
      "response": "Absolutely! With companies like SpaceX working on Mars colonization, humans may one day call other planets home. It’s just a matter of time and technological progress."
    },
    {
      "prompt": "What jobs will exist in the future?",
      "response": "Future jobs might include AI ethicists, space tourism guides, climate engineers, memory implant designers, and even asteroid miners. Creativity and problem-solving will remain key skills."
    },
    {
      "prompt": "Can you predict the future?",
      "response": "I can’t predict the future, but I can help you imagine possibilities! Whether it’s flying cars or curing diseases, the future depends on choices we make today."
    }
    ]
    
    with open('sample.jsonl','w') as f:
        for item in sample_data:
            json.dump(item,f)
            f.write('\n')
    
    prompts = [item['prompt'] for item in sample_data]
    responses = [item['response'] for item in sample_data]
    
df = pd.DataFrame({'prompt':prompts, 'response':responses})

print('Dataset Size: ',len(df))   
print('Sample Prompt and responses: ')
print(df.head())

prompt_lenghts = df['prompt'].apply(lambda x: len(x.split())) 
print("Average prompt lenght:",prompt_lenghts.mean())
print("max prompt lenght: ",prompt_lenghts.max())

all_words = ''.join(df["prompt"]).split()
word_counts = Counter(all_words)
print("Top 5 Common Words: ",word_counts.most_common(5))

        