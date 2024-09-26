from query_interface import query_ent_name,query_1hop_relation,query_2hop_relation,get_1hop_chain,get_2hop_chain,get_2hop_triples
import json
import csv
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
from sim_compute import Similarity

DATA='GraphQuestions'
K=100

def has_digit(input_string):
    for char in input_string:
        if char.isdigit():
            return True
    return False
    
def replace_entity(string, entity, replacement):
    pattern = re.compile(re.escape(entity), re.IGNORECASE)
    new_string = re.sub(pattern, replacement, string)
    return new_string

# construct name dict
id2name = {}
for file_name in tqdm(os.listdir('../../Freebase/id2name_parts')):
    with open(os.path.join('../../Freebase/id2name_parts', file_name), 'r') as rf:
        data_input = csv.reader(rf, delimiter="\t")
        for row in data_input:
            id2name[row[0]] = row[2]
            
def convert_names(triples):
    triplelist=[]
    for t in triples:
        # literal
        if t[0][0:2] not in ['m.','n.','g.']:
            head=t[0].replace('-08:00','') 
        # entity
        else:
            if id2name.get(t[0]):
                head=id2name[t[0]].replace('-08:00','')
            else:
                temp=query_ent_name(t[0])
                if temp:
                    head=temp.replace('-08:00','')
                else:
                    head=t[0]
        # literal
        if t[2][0:2] not in ['m.','n.','g.']:
            tail=t[2].replace('-08:00','') 
        # entity
        else:
            if id2name.get(t[2]):
                tail=id2name[t[2]].replace('-08:00','')
            else:
                temp=query_ent_name(t[2])
                if temp:
                    tail=temp.replace('-08:00','')
                else:
                    tail=t[2]     
        triplelist.append('('+head+', '+t[1]+', '+tail+')')
    return triplelist

# similarity model
sim_model=Similarity()

retrieve_subgraph=[]
# retrieve metrics
accuracy=0
recall=0
# query neighbour 
dataset=json.load(open('../../../../subgraph/'+DATA+'/data/test.json','r',encoding='utf-8'))
for index,sample in tqdm(enumerate(dataset)):
    if DATA in ['grailqa','GraphQuestions']:
        # question, entity, head, answer
        question=sample["question"]
        entity=[]
        head=[]
        for n in sample["graph_query"]["nodes"]:
            if n["node_type"]=="entity":
                entity.append(n["friendly_name"])
                head.append(n["id"])
                id2name[n["id"]]=n["friendly_name"]
        # replace entity surface to [MASK]
        question_mask=question
        for i in entity:
            question_mask=replace_entity(question_mask,i,'[MASK]')
        answer=[]
        for i in sample["answer"]:   
            if i.get("entity_name"):
                answer.append(i["entity_name"])
                id2name[i["answer_argument"]]=i["entity_name"]
            else:
                answer.append(i["answer_argument"]) 
    if DATA in ['WebQSP']:
        # question, entity, head, answer
        question=sample["question"]
        entity=[sample["headname"]]
        question_mask=replace_entity(question,entity[0],'[MASK]')
        head=[sample["headmid"]]
        answer=sample["answername"].split('|')
        id2name[head[0]]=entity[0]
        for ansmid,ans in zip(sample["answermid"].split('|'),sample["answername"].split('|')):
            id2name[ansmid]=ans
    # if no head entity, no retrieve result
    if len(head)==0:
        graphdict=dict()
        graphdict['question']=question
        graphdict['triples']=[]
        graphdict['answers']=answer
        retrieve_subgraph.append(graphdict)
        continue
    # query 1 hop and 2 hop relation
    reset=set()
    en2rel=dict()
    for ent in head:
        rel_1hop=query_1hop_relation(ent)
        chain_2hop=query_2hop_relation(ent)
        rel_2hop=set()
        for i in chain_2hop:
            rel_2hop.add(i[0]+' '+i[1])
        en2rel[ent]=rel_1hop|rel_2hop
        reset=reset|rel_1hop
        reset=reset|rel_2hop
    # if no relation, no retrieve result
    if len(reset)==0:
        graphdict=dict()
        graphdict['question']=question
        graphdict['triples']=[]
        graphdict['answers']=answer
        retrieve_subgraph.append(graphdict)
        continue
    # select most similar relation chain
    sort_re=sim_model.compute(question,list(reset))
    # sample from KG
    triples=[]
    for rechain in sort_re:
        temp_h=[]
        for ent,rel in en2rel.items():
            if rechain in rel:
                temp_h.append(ent)
        if ' ' in rechain:
            for h in temp_h:
                triples.extend(convert_names(get_2hop_chain(h,rechain.split(' '))))
        else:
            for h in temp_h:
                triples.extend(convert_names(get_1hop_chain(h,rechain)))
        if len(set(triples))>=K:
            break
    # avoid redundant triples
    triples1=[]
    for i in triples[:K]:
        if i not in triples1:
            triples1.append(i)
    contents=' '.join(triples1[:50])
    # calculate retrieve metrics
    FLAG=False
    temp_r=0
    for a in answer:
        if a.lower() in contents.lower():
            FLAG=True
            temp_r+=1
    if FLAG:
        accuracy+=1
        recall+=temp_r/len(answer)
    graphdict=dict()
    graphdict['question']=question
    graphdict['triples']=triples1
    graphdict['answers']=answer
    retrieve_subgraph.append(graphdict)
    print('*'*30,'Current Retrieve Results','*'*30)
    print('Accuracy:',accuracy/(index+1))
    print('Recall:',recall/(index+1))

# save retrieve results
os.makedirs('results', exist_ok=True)
json.dump(retrieve_subgraph,open('results/'+DATA+'.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print('*'*30,'Retrieve Results','*'*30)
print('Accuracy:',accuracy/len(dataset))
print('Recall:',recall/len(dataset))