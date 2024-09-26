import json
import os
import random


DATA='GraphQuestions'
# llm: Meta-Llama-3-8B-Instruct, llama-2-7b-chat-hf
LLM='llama-2-7b-chat-hf'

data=json.load(open(DATA+'/PA-Mistral/CoT/'+LLM+'/sample.json','r',encoding='utf-8'))

ablation=[]

kr_prompt='''Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: {triple}
Question: {ques}
'''

for sample in data:
    if sample['source']=='single knowledge':
        continue
    if len(sample['prefer'])!=0:
        verdict=sample['prefer'].split('\n')[-1]
        if verdict=='A':
            temp=dict()
            temp['prompt']=kr_prompt.format(triple=sample["noisy"],ques=sample["question"])
            temp['chosen']=sample["output_list"][0]
            temp["rejected"]=sample["output_list"][1]
            ablation.append(temp)
        if verdict=='B':
            temp=dict()
            temp['prompt']=kr_prompt.format(triple=sample["noisy"],ques=sample["question"])
            temp['chosen']=sample["output_list"][1]
            temp["rejected"]=sample["output_list"][0]
            ablation.append(temp)        
    else:
        num1=0
        num2=0
        for a in sample["answer"]:
            if a in sample["output_list"][0]:
                num1+=1
            if a in sample["output_list"][1]:
                num2+=1    
        if num1>num2:
            temp=dict()
            temp['prompt']=kr_prompt.format(triple=sample["noisy"],ques=sample["question"])
            temp['chosen']=sample["output_list"][0]
            temp["rejected"]=sample["output_list"][1]
            ablation.append(temp)
        if num1<num2:     
            temp=dict()
            temp['prompt']=kr_prompt.format(triple=sample["noisy"],ques=sample["question"])
            temp['chosen']=sample["output_list"][1]
            temp["rejected"]=sample["output_list"][0]
            ablation.append(temp)   

print(len(ablation))
os.makedirs(DATA+'/PA-Mistral-ablation/CoT/'+LLM,exist_ok = True)
random.shuffle(ablation)
train_num=int(len(ablation)*0.9)
json.dump(ablation[:train_num],open(DATA+'/PA-Mistral-ablation/CoT/'+LLM+'/train.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
json.dump(ablation[train_num:],open(DATA+'/PA-Mistral-ablation/CoT/'+LLM+'/dev.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)