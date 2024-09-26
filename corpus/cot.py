import json
import math
import copy
import time
import random
from openai import OpenAI
from tqdm import tqdm
import os

client=OpenAI(api_key='YOUR KEY')

interval=200
DATA='GraphQuestions'
# set EX_RATE
if DATA in ['GraphQuestions','WebQSP']:
    EX_RATE=1
if DATA in ['grailqa']:
    EX_RATE=0.5
    
train=json.load(open('../subgraph/'+DATA+'/graph/train.json','r',encoding='utf-8'))

os.makedirs(DATA+'/finetune/'+DATA+'/CoT/train/',exist_ok=True)
os.makedirs(DATA+'/finetune/'+DATA+'/CoT/middle/',exist_ok=True)

kr_prompt='''Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: (Oxybutynin Oral, medicine.routed_drug.route_of_administration, Oral administration) (Oxybutynin Oral, medicine.routed_drug.marketed_formulations, Oxybutynin chloride 5 extended release film coated tablet) (Oxybutynin Chloride Oral, medicine.routed_drug.marketed_formulations, Oxybutynin chloride 5 extended release film coated tablet) (Oxybutynin chloride 5 extended release film coated tablet, medicine.drug_formulation.formulation_of, Oxybutynin)
Question: oxybutynin chloride 5 extended release film coated tablet is the ingredients of what routed drug?
Reason 1: I need to know which routed drug contains oxybutynin chloride 5 extended release film coated tablet.
Knowledge 1: "Oxybutynin Chloride Oral" is a type of routed drug and "Oxybutynin chloride 5 extended release film coated tablet" is one of the marketed formulations of "Oxybutynin Chloride Oral".

Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: (Google, organization.organization.founders, Sergey Brin) (Sergey Brin, people.person.education, CVT1) (CVT1, education.education.institution, University of Maryland, College Park) (Google, organization.organization.founders, Larry Page) (Larry Page, people.person.education, CVT1) (CVT1, education.education.institution, University of Michigan) (CVT1, education.education.institution, Stanford University)
Question: where did the founder of google go to college?
Reason 1: I need to know who is the founder of google.
Knowledge 1: Sergey Brin and Larry Page is the founder of google.
Reason 2: I need to know Sergey Brin and Larry Page go to which the university.
Knowledge 2: Sergey Brin studied at the University of Maryland, College Park and Stanford University. Larry Page studied at the University of Michigan and Stanford University.

Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: (Rock music, music.genre.artists, Outkast) (Rock music, music.genre.parent_genre, Folk music) (Rock music, music.genre.albums, The Confessions Tour) (Electronica, music.genre.artists, Bright Eyes) (Electronica, music.genre.parent_genre, House music) (Electronica, music.genre.albums, The Confessions Tour) (Electronica, music.genre.artists, t.A.T.u.)
Question: the albums confessions tour is part of what parent genre of a musical genre?
Reason 1: I need to know the musical genre of the albums confessions tour.
Knowledge 1: The album confessions tour is associated with the genre Rock music and Electronica.
Reason 2: I need to know the parent genre of Rock music and Electronica.
Knowledge 2: The parent genre of Rock music is Folk music. The parent genre of Electronica is House music.

Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: {triple}
Question: {ques}
'''

kr_prompt1='''Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: {triple}
Question: {ques}
'''

ans_prompt='''Below are the facts that might be relevant to answer the question:
{knowledge}
Question: {ques}
Answer:'''

num_dict = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

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

data=[]
resume=0
#data=json.load(open('train-'+str(resume)+'.json','r',encoding='utf-8'))
index=resume
for sample in tqdm(train[resume:]):
    index+=1
    # gold graph
    gold_g=set()
    for i in sample['restrict_graph']:
        for j in i:
            temp='('+j[0]+', '+j[1]+', '+j[2]+')'
            gold_g.add(temp)
    # shuffle gold graph
    gold_g=list(gold_g)
    random.shuffle(gold_g)
    
    # extend graph
    extend=set()
    for i in sample["ex_graph"]:
        for j in i:
            temp='('+j[0]+', '+j[1]+', '+j[2]+')'
            if temp not in gold_g:
                extend.add(temp)
    extend=list(extend)
    random.shuffle(extend)
    
    # extend number filter
    ex_filter=set()
    NUM=math.ceil(len(gold_g)*EX_RATE) 
    # first use no CVT triple
    for i in extend:
        if 'CVT' not in i:
            ex_filter.add(i)
        if len(ex_filter)==NUM:
            break
    # add CVT triple
    if len(ex_filter)<NUM: 
        for i in extend:
            if 'CVT' in i:
                ex_filter.add(i)
            if len(ex_filter)==NUM:
                break      
    
    # noisy graph
    noisy=set(gold_g).union(ex_filter)
    # random shuffle
    noisy=list(noisy)
    random.shuffle(noisy)
    # noisy graph string
    noisy_string=''
    for i in noisy:
        noisy_string=noisy_string+i+' '
    
    # data generation
    # knowledge rewriter
    knowledge=getResponse(kr_prompt.format(triple=noisy_string.strip(),ques=sample["question"]))
    print(kr_prompt.format(triple=noisy_string.strip(),ques=sample["question"]))
    print(knowledge)
    # extract knowledge
    know=''
    for line in knowledge.split('\n'):
        if not line.startswith('Reason') and len(line.strip())!=0:
            if 'Knowledge ' in line:
                know=know+line.replace('Knowledge ','')[3:]+'\n'
            else:
                know=know+line+'\n'
    # format knowledge
    format_know=''
    for line in knowledge.split('\n'):
        if line.startswith('Knowledge') or line.startswith('Reason'):
            format_know=format_know+line+'\n'
    # skip null format knowledge
    if len(format_know.strip())==0:
        continue
    # knowledge augmented response
    answer=getResponse(ans_prompt.format(knowledge=know.strip(),ques=sample["question"]))
    print(ans_prompt.format(knowledge=know.strip(),ques=sample["question"]))
    print(answer)
    
    # gold answer extraction
    if DATA=='WebQSP':
        gold=sample["answer"]
    else:
        gold=[]
        for i in sample["answer"]:
            if i.get("entity_name"):
                gold.append(i["entity_name"])
            else:
                if i["answer_argument"].isdigit() and num_dict.get(i["answer_argument"]):
                    gold.append(num_dict[i["answer_argument"]])
                else:
                    gold.append(i["answer_argument"])

    # result
    result=''
    FLAG=True
    for i in gold:
        if i.lower() not in answer.lower():
            FLAG=False
            break
    if FLAG:
        result='correct'
    else:
        result='incorrect'
    
    # record
    if FLAG:
        samdict=dict()
        samdict['question']=sample['question']
        samdict['graph']=list(gold_g)
        samdict['ex_graph']=noisy
        samdict['know_prompt']=kr_prompt1.format(triple=noisy_string.strip(),ques=sample["question"])
        samdict['knowledge']=format_know.strip()
        samdict['answer']=gold
        samdict['response']=answer
        data.append(samdict)
    if index%interval==0:
        json.dump(data,open(DATA+'/finetune/'+DATA+'/CoT/middle/all-'+str(index)+'.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)

# save
random.shuffle(data)
json.dump(data,open(DATA+'/finetune/'+DATA+'/CoT/all.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)

# convert data to specific template
train_num=int(len(data)*0.9)
train=[]
dev=[]
for i in data[:train_num]:
    temp=dict()
    temp["instruction"]=i['know_prompt']
    temp["input"]=''
    temp["output"]=i['knowledge']
    train.append(temp)
    
for i in data[train_num:]:
    temp=dict()
    temp["instruction"]=i['know_prompt']
    temp["input"]=''
    temp["output"]=i['knowledge']
    dev.append(temp)
    
json.dump(train,open(DATA+'/finetune/'+DATA+'/CoT/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(dev,open(DATA+'/finetune/'+DATA+'/CoT/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
