import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random
from tqdm import tqdm
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer,AutoModel
import torch
from peft import PeftModel
import sys
import openai
import time
from openai import OpenAI

# generation config
generation_config = GenerationConfig(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=256
)

# dataset: grailqa, GraphQuestions
DATA='grailqa'
# chatgpt, Mistral-7B-Instruct-v0.3
ANS='chatgpt'

# path for LLM
LLM_PATH='../../../../pretrain/'+ANS
# path for tokenizer
TOKENIZER_PATH='../../../../pretrain/'+ANS

if ANS!='chatgpt':
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    llm=AutoModelForCausalLM.from_pretrained(LLM_PATH,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='cuda:0')

# set client
client=OpenAI(api_key='YOUR KEY')

test=json.load(open('../retrieve/2hop/format/'+DATA+'.json','r',encoding='utf-8'))

ans_prompt='''Question: {ques}
Answer: '''

num_dict = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
}
   
# result
result='result/'+ANS+'/'+DATA+'/no_knowledge.json'
os.makedirs('result/'+ANS+'/'+DATA,exist_ok = True)
log_file='log/'+ANS+'/'+DATA+'/no_knowledge.log'
os.makedirs('result/'+ANS+'/'+DATA,exist_ok = True)

# redirect output to log
sys.stdout = open(log_file, 'w')

def getResponse(prompt,max_retries=10):
    # set retries
    retries=0
    while retries < max_retries:
        try:
            res = client.chat.completions.create(
                model='gpt-3.5-turbo',
                #model='gpt-4',
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 1 minutes...")
            retries += 1
            time.sleep(60)
    return ''

def LLMResponse(prompt,llm,tokenizer,cuda):
    inputs = tokenizer(prompt,return_tensors="pt")
    generation_output = llm.generate(
            input_ids=inputs["input_ids"].to(cuda),
            attention_mask=inputs['attention_mask'].to(cuda),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config
        )
    output = tokenizer.decode(generation_output[0],skip_special_tokens=True)
    response = output.split(prompt)[-1].strip()
    return response

index=0
f1=0
acc=0
EM=0
data=[]
for sample in tqdm(test):
    index+=1
    
    # response
    if ANS!='chatgpt':
        answer=LLMResponse(ans_prompt.format(ques=sample["question"]),llm,tokenizer,'cuda:0')
    else:
        answer=getResponse(ans_prompt.format(ques=sample["question"]))
    print(ans_prompt.format(ques=sample["question"]))
    print(answer)
    
    # gold answer extraction
    gold=sample["answer"]
    
    # gold number
    gold_num=[]
    for i in gold:
        if i.isdigit() and num_dict.get(i):
            gold_num.append(num_dict[i])

    # result
    cor=0
    FLAG=False
    FLAG1=True
    FLAG2=False
    # judge use gold_num or gold
    for i in gold_num:
        if i.lower() in answer.lower():
            FLAG2=True
            break
    if FLAG2:
        gold_ans=gold_num
    else:
        gold_ans=gold
    if len(answer)!=0:
        for i in gold_ans:
            if i.lower() in answer.lower():
                FLAG=True
                cor+=1
            if i.lower() not in answer.lower():
                FLAG1=False      
        if FLAG:
            acc+=1
        f1+=cor/len(gold_ans)
        if FLAG1:
            EM+=1    
    else:
        FLAG=False
        FLAG1=False
        cor=0

    # record
    temp=dict()
    temp['question']=sample['question']
    temp['answer']=gold_ans
    temp['response']=answer
    if FLAG:
        temp['accuracy']=1
    else:
        temp['accuracy']=0
    temp['f1']=cor/len(gold)
    if FLAG1:
        temp['EM']=1
    else:
        temp['EM']=0
    data.append(temp)
    print('Current Accuracy: {}'.format(acc/index))
    print('Current F1: {}'.format(f1/index))
    print('Current EM: {}'.format(EM/index))
    sys.stdout.flush()

print('Accuracy: {}'.format(acc/len(test)))
print('F1: {}'.format(f1/len(test)))
print('EM: {}'.format(EM/len(test)))
json.dump(data,open(result,'w',encoding='utf-8'),indent=2,ensure_ascii=False)
