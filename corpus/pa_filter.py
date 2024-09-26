import json
import random
from transformers import AutoTokenizer
from tqdm import tqdm

# grailqa, GraphQuestions
DATA='grailqa'
# llama-2-7b-chat-hf, bloom-3b, Meta-Llama-3-8B-Instruct
LLM='Meta-Llama-3-8B-Instruct'

data=json.load(open(DATA+'/PA-Mistral/CoT/'+LLM+'/middle/DPO-8000.json','r',encoding='utf-8'))
tokenizer = AutoTokenizer.from_pretrained('../../pretrain/'+LLM)

def detect_repeated_text(text, threshold=5):
    words = text.split()
    word_count = {}

    for word in words:
        word_count[word] = word_count.get(word, 0) + 1

    repeated_words = [word for word, count in word_count.items() if count > threshold]
    
    if len(repeated_words)>threshold:
        return True
    else:
        return False
    
    return repeated_words

# filter too long sequence
data1=[]
num=0
for sample in tqdm(data):
    
    p_l=len(tokenizer(sample['prompt'],return_tensors="pt")["input_ids"][0])
    c_l=len(tokenizer(sample['chosen'],return_tensors="pt")["input_ids"][0])
    r_l=len(tokenizer(sample['rejected'],return_tensors="pt")["input_ids"][0])
    t_l=max(c_l,r_l)
    s_l=p_l+t_l
    
    if p_l>1024 or c_l>512 or r_l>1024 or s_l>2048:
        continue
        
    chosen=sample['chosen'].strip().split('\n')
    if len(chosen)>6:
        continue
    FLAG=True
    for index,line in enumerate(chosen):
        if index%2==0:
            if not line.startswith('Reason '):
                FLAG=False
                break
        else:
            if not line.startswith('Knowledge '):
                FLAG=False
                break            
    if FLAG:
        data1.append(sample)
    
    #if detect_repeated_text(sample['chosen']):
    #    print(sample)
    
print(len(data1))

# divide into train and dev
random.shuffle(data1)
train_num=int(len(data1)*0.9)
json.dump(data1[:train_num],open(DATA+'/PA-Mistral/CoT/'+LLM+'/train.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
json.dump(data1[train_num:],open(DATA+'/PA-Mistral/CoT/'+LLM+'/dev.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
