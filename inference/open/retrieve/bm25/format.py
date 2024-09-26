import json
import os
import re

DATA='grailqa'
NUM=20

def has_digit(input_string):
    for char in input_string:
        if char.isdigit():
            return True
    return False

with open('../../../../subgraph/'+DATA+'/data/test.json', 'r') as rf:
    data = json.load(rf)
    
with open('results/'+DATA+'.json', 'r') as rf:
    documents = json.load(rf)

result=[]
for sample,doc in zip(data,documents):
    if DATA in ['WebQSP']:
        question=sample["question"]
        answer=sample["answername"].split('|')
    if DATA in ['GraphQuestions','grailqa']:
        question=sample["question"]
        answer=[]
        for i in sample["answer"]:   
            if i.get("entity_name"):
                answer.append(i["entity_name"])
            else:
                answer.append(i["answer_argument"])
    doclist=doc["documents"]
    triple_str=''
    for d in doclist:
        triple_str=triple_str+d["triples"]+' '
    triple_str=triple_str[:-1]
    # avoid redundant triples
    triplelist=triple_str.split(') (')
    triplelist[0]=triplelist[0][1:]
    triplelist[-1]=triplelist[-1][:-1]
    triplelist1=[]
    for i in triplelist:
        if len(i)>100:
            continue
        if len(i.split(', '))<3:
            continue
        rel=i.split(', ')[1]
        # skip relations
        if rel.startswith('common') or rel.startswith('type.object') or rel.startswith('freebase') or rel.endswith('type') or rel.endswith('label'):
            continue
        #print(i.split(', '))
        # skip triples with too long object
        if i not in triplelist1:
            triplelist1.append(i)
    # construct triple string
    triples=''
    for i in triplelist1[:NUM]:
        triples=triples+'('+i+') '
    triples=triples[:-1]
    # convert mid to cvt
    candidate=re.findall(r'm\.[\da-zA-Z_]+', triples)
    candidate.extend(re.findall(r'g\.[\da-zA-Z_]+', triples))
    candidate.extend(re.findall(r'n\.[\da-zA-Z_]+', triples))
    cvtmid=[]
    cvt_num=1
    for i in candidate:
        if has_digit(i):
            if i not in cvtmid:
                cvtmid.append(i)
    for i in cvtmid:
        triples=triples.replace(i,'CVT'+str(cvt_num))
        cvt_num+=1
    samdict=dict()
    samdict["question"]=question
    samdict["answer"]=answer
    samdict["triples"]=triples
    result.append(samdict)

json.dump(result,open('format/'+DATA+'.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)