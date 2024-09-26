import os
import json
import math
import random
from tqdm import tqdm

# dataset: grailqa, GraphQuestions
DATA='grailqa'

# result for subgraph
result=DATA+'/gold/test.json'

# load data
data=json.load(open(DATA+'/graph/test.json','r',encoding='utf-8'))

num_dict = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
}

MAX_NUM=10

samplelist=[]
for sample in tqdm(data):
    # graph sample
    graphset=set()
    for i in sample['graph'][:MAX_NUM]:
        for j in i:
            graphset.add('('+j[0]+', '+j[1]+', '+j[2]+')')
    # avoid too many triples
    graphlist=list(graphset)
    

    # gold answer extraction
    if DATA=='WebQSP':
        gold=sample["answer"]
    else:
        gold=[]
        for i in sample["answer"]:
            if i.get("entity_name"):
                gold.append(i["entity_name"])
            else:
                gold.append(i["answer_argument"])

   
    # save
    temp=dict()
    temp['question']=sample['question']
    temp["triples"]=' '.join(graphlist)
    temp['answer']=gold
    samplelist.append(temp)

json.dump(samplelist,open(result,'w',encoding='utf-8'),indent=2,ensure_ascii=False)
    