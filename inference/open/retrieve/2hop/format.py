import json
import os
import re

# grailqa, GraphQuestions, WebQSP
DATA='grailqa'
NUM=30

def has_digit(input_string):
    for char in input_string:
        if char.isdigit():
            return True
    return False
    
with open('results/'+DATA+'.json', 'r') as rf:
    documents = json.load(rf)

accuracy=0
recall=0
result=[]
for doc in documents:
    question=doc["question"]
    answer=doc["answers"]
    # avoid redundant triples
    triplelist=doc["triples"]
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
    triples=' '.join(triplelist1[:NUM])
    # convert mid to cvt
    candidate=re.findall(r'm\.[\da-zA-Z_]+', triples)
    candidate.extend(re.findall(r'g\.[\da-zA-Z_]+', triples))
    candidate.extend(re.findall(r'n\.[\da-zA-Z_]+', triples))
    cvtmid=[]
    for i in candidate:
        if has_digit(i):
            if i not in cvtmid:
                cvtmid.append(i)
    cvt_num=1
    for i in cvtmid:
        triples=triples.replace(i,'CVT'+str(cvt_num))
        cvt_num+=1
    samdict=dict()
    samdict["question"]=question
    samdict["answer"]=answer
    samdict["triples"]=triples
    result.append(samdict)
    FLAG=False
    r=0
    for i in answer:
        if i in triples:
            FLAG=True
            r+=1
    if FLAG:
        accuracy+=1
        recall+=r/len(answer)

json.dump(result,open('format/'+DATA+'.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print('Accuracy:',accuracy/len(result))
print('Recall:',recall/len(result))
