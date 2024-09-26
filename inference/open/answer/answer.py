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
# rewrite llm: llama-2-7b-chat-hf, Meta-Llama-3-8B-Instruct, chatgpt
LLM='Meta-Llama-3-8B-Instruct'
# answer llm: Mistral-7B-Instruct-v0.3, chatgpt
ANS='Mistral-7B-Instruct-v0.3'
# retrieve method: bm25, 2hop
MODE='2hop'
# knowledge representation: pa-chatgpt, pa-mistral, chain, summary, text, triple
KR='chain'

# load different knowledge representation
if KR=='triple':
    test=json.load(open('../retrieve/'+MODE+'/format/'+DATA+'.json','r',encoding='utf-8'))
else:
    test=json.load(open('../rewrite/result/'+DATA+'/'+MODE+'/'+LLM+'/'+KR+'.json','r',encoding='utf-8'))
    
# set client
client=OpenAI(api_key='YOUR KEY')

ans_chain_prompt='''Your task is to answer the question based on the reasoning chain that might be relevant. Try to use the original words from the given knowledge to answer the question. But if it is not useful, just ignore it and generate your own guess.
{knowledge}
Question: {ques}
Answer: '''

ans_prompt='''Your task is to answer the question based on the knowledge that might be relevant. Try to use the original words from the given knowledge to answer the question. But if it is not useful, just ignore it and generate your own guess.
{knowledge}
Question: {ques}
Answer: '''

num_dict = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
}
   
# path for ANS
LLM_PATH='../../../../pretrain/'+ANS
# path for tokenizer
TOKENIZER_PATH='../../../../pretrain/'+ANS
# graph
origin=json.load(open('../../../subgraph/'+DATA+'/gold/test.json','r',encoding='utf-8'))
# result
if KR=='triple':
    result='result/'+ANS+'/'+DATA+'/'+MODE+'/triple.json'
    os.makedirs('result/'+ANS+'/'+DATA+'/'+MODE,exist_ok = True)
    log_file='log/'+ANS+'/'+DATA+'/'+MODE+'/triple.log'
    os.makedirs('log/'+ANS+'/'+DATA+'/'+MODE,exist_ok = True)
else:
    result='result/'+ANS+'/'+DATA+'/'+MODE+'/'+LLM+'/'+KR+'.json'
    os.makedirs('result/'+ANS+'/'+DATA+'/'+MODE+'/'+LLM,exist_ok = True)
    log_file='log/'+ANS+'/'+DATA+'/'+MODE+'/'+LLM+'/'+KR+'.log'
    os.makedirs('log/'+ANS+'/'+DATA+'/'+MODE+'/'+LLM,exist_ok = True)

# redirect output to log
sys.stdout = open(log_file, 'w')

if ANS!='chatgpt':
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    llm=AutoModelForCausalLM.from_pretrained(LLM_PATH,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='cuda:0')

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

def format(knowledge):
    # format knowledge
    if knowledge.startswith('Your task is to '):
        # format knowledge: chain, pa
        if 'Question: ' in knowledge:
            knowledge=knowledge.split('\n')
            knowledge='\n'.join(knowledge[3:])
    
        # format knowledge: summary
        if 'Knowledge: ' in knowledge:
            knowledge=knowledge.split('Knowledge: ')[1]
        
        # format knowledge: text
        if 'The sentence is: ' in knowledge:
            knowledge=knowledge.split('The sentence is: ')[1]
    return knowledge

index=0
f1=0
acc=0
EM=0
data=[]
for sample,sample1 in tqdm(zip(origin,test),total=len(origin)):
    index+=1
    if KR=='triple':
        knowledge=sample1["triples"]
    else:
        knowledge=sample1["knowledge"]
    
    # format knowledge
    knowledge=format(knowledge)
    
    # knowledge augmented response
    if KR in ['pa-chatgpt','pa-mistral','chain']:
        prompt=ans_chain_prompt.format(knowledge=knowledge.strip(),ques=sample["question"])
    else:
        prompt=ans_prompt.format(knowledge=knowledge.strip(),ques=sample["question"])

    if ANS!='chatgpt':
        answer=LLMResponse(prompt,llm,tokenizer,'cuda:0')
    else:
        answer=getResponse(prompt)
    print(prompt)
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
    temp['graph']=sample["triples"]
    temp['knowledge']=knowledge
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
