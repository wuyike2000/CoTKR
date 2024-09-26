import json
import re
from sparql_utils.sparql_executor import execute_query, execute_query_allvar, get_friendly_name
from query_interface import get_1hop
from tqdm import tqdm
import random
import copy

# max extend triple number for each entity in gold graph
EXNUM=10
# max graph number for extend
GRAPHNUM=10

# change sparql to query all variables
def update_sparql_query(query_string):
    # find all ver
    var_pattern = r'\?[xy]\d+'
    variables = set(re.findall(var_pattern, query_string))
    # sort based on num
    variables = sorted(variables, key=lambda x: int(x[2:]))
    # extract triples with y for it may not be used in main query
    query_lines = query_string.split('\n')
    query_y=[]
    for line in query_lines:
        if line.startswith("?") and len(line.split(' ')) == 5 and '?y' in line:
            query_y.append(line)
    query_lines=query_lines[:3]+query_y+query_lines[3:]
    # modify select distinct
    if query_lines[1].startswith('SELECT '):
        # remove SELECT (?x0 AS ?value) WHERE { and last }
        query_lines = query_lines[:1] + query_lines[2:-1]
        if query_lines[1].startswith('SELECT DISTINCT'):
            select_parts = query_lines[1].split(' ')
            select_parts[2] = ' '.join(variables)
            query_lines[1] = ' '.join(select_parts)
    return '\n'.join(query_lines), list(variables)
    
# parse sparql to subgraph
def sparql_to_graph(query):
    # input: sparql query
    # return: str(triples), 
    lines = query.split('\n')
    graph_lines = []
    values = {}
    # extract all intermediate entity mid
    for line in lines:
        if line.startswith("VALUE"):
            k = None
            v = None
            for item in line.split(' '):
                if item.startswith("?"):
                    k = item
                if item.startswith(":") or "<http://www.w3.org/2001/XMLSchema#" in item:
                    v = item
            assert k != None and v != None
            values[k] = v
    
    # extract triples from sparql query
    for line in lines:
        if line.startswith("SELECT DISTINCT"): # make sure Answer always is ?x0
            assert line == 'SELECT DISTINCT ?x0  WHERE { '
        # only process triple lines
        if line.startswith("?") and len(line.split(' ')) == 5:
            graph_lines.append(line)
    graph = '\n'.join(graph_lines)
    return graph
    
def query(file_type):
    file_name = 'data/' + file_type +'.json'
    
    data = json.load(open(file_name, 'r'))
    mid_dict = {}
    graphdata=[]
    
    for one_example in tqdm(data):
        # one sample
        sample=dict()
        # cvt dict, for one sample
        cvt_dict=dict()
        # cvt index
        cvt=1
        # sparql query
        sparql_query = one_example['sparql_query']
        sparql_query, variables= update_sparql_query(sparql_query)
        res = execute_query_allvar(sparql_query)
        # name query
        name=[]
        if res != None:
            for one_combo in res:
                # one name
                one_name=dict()
                # iterate query results
                for k,v in one_combo.items():
                    value = 'null'
                    one_combo[k] = {'mid': v['value'], 'value': v['value']}
                    if v['value'].startswith('http://rdf.freebase.com/ns/'):
                        mid = v['value'].replace('http://rdf.freebase.com/ns/', '').replace('-08:00','')
                        if mid in mid_dict:
                            value = mid_dict[mid]
                        else:
                            try:
                                value = get_friendly_name(mid)
                                if value!='null':
                                    mid_dict[mid] = value
                            except:
                                print(mid)
                    else:
                        mid=v['value'].replace('-08:00','')
                        value=v['value'].replace('-08:00','')
                    if value!='null': 
                        one_name[k] = {'mid': mid, 'value': value}
                    else:
                        if cvt_dict.get(mid):
                            one_name[k] = {'mid': mid, 'value': cvt_dict[mid]}
                        else:
                            one_name[k] = {'mid': mid, 'value': 'CVT'+str(cvt)}
                            cvt_dict[mid]='CVT'+str(cvt)
                            cvt+=1    
                name.append(one_name)
        
        # extract entity mid from graph
        midlist=[]
        # graph name exchange
        graph=[]
        graphstr=sparql_to_graph(one_example['sparql_query'])
        graphstr=graphstr.replace(' . ','').split('\n')
        for n in name:
            midset=set()
            # one subgraph for ques
            one_graph=[]
            # mid to name
            for i in graphstr:
                triple=[]
                j=i.split(' ')
                for k in j:
                    if k.startswith('?'):
                        triple.append(n[k[1:]]['value'])
                        continue
                    if k.startswith(':'):
                        triple.append(k[1:])
                        continue
                    triple.append(k)
                # add mid
                # j[0]
                # make sure j[0] is an entity
                if j[0].startswith('?') and len(n[j[0][1:]]['mid'])>1 and n[j[0][1:]]['mid'][0:2] in ['m.','n.','g.']:
                    midset.add(n[j[0][1:]]['mid'])
                # j[2]
                # make sure j[2] is an entity
                if j[2].startswith('?') and len(n[j[2][1:]]['mid'])>1 and n[j[2][1:]]['mid'][0:2] in ['m.','n.','g.']:
                    midset.add(n[j[2][1:]]['mid'])
                # skip type relation               
                if triple[1]!='type.object.type':
                    one_graph.append(triple)
            midlist.append(midset)
            random.shuffle(one_graph)
            graph.append(one_graph)
        
        # graph extend
        ex_graph=[]
        for index,g in enumerate(graph[:GRAPHNUM]):
            # copy g to g1
            g1=copy.deepcopy(g)
            # iteratively extend triple
            ex_triple=[]
            # collect mid triple
            mid_triple=[]
            for j in midlist[index]:
                for k in get_1hop(j)[:EXNUM]:
                    if k not in mid_triple:
                        mid_triple.append(k)
            # avoid redundant triple
            unique_triples = set(tuple(triple) for triple in mid_triple)
            mid_triple = [list(triple) for triple in unique_triples]
            random.shuffle(mid_triple)
            # mid to name
            for k in mid_triple:
                extend=[]
                # k[0]
                temp=''
                # k[0] is in mid_dict
                if mid_dict.get(k[0]): 
                    temp=mid_dict[k[0]]
                # k[0] is not entity
                if len(temp)==0 and (len(k[0])==1 or k[0][0:2] not in ['m.','n.','g.']):
                    temp=k[0].replace('-08:00','')
                # k[0] is entity                    
                if len(temp)==0:
                    temp=get_friendly_name(k[0])
                    if temp=='null':
                        if cvt_dict.get(k[0]):
                            temp=cvt_dict[k[0]]
                        else:
                            temp='CVT'+str(cvt)
                            cvt_dict[k[0]]=temp
                            cvt+=1
                    else:
                        temp=temp.replace('-08:00','')
                extend.append(temp)
                # k[1]
                extend.append(k[1])
                # k[2]                    
                temp='' 
                # k[2] is in mid_dict
                if mid_dict.get(k[2]):
                    temp=mid_dict[k[2]]
                # k[2] is not entity
                if len(temp)==0 and (len(k[2])==1 or k[2][0:2] not in ['m.','n.','g.']):
                    temp=k[2].replace('-08:00','') 
                # k[2] is entity                   
                if len(temp)==0:
                    temp=get_friendly_name(k[2])
                    if temp=='null':
                        if cvt_dict.get(k[2]):
                            temp=cvt_dict[k[2]]
                        else:
                            temp='CVT'+str(cvt)
                            cvt_dict[k[2]]=temp
                            cvt+=1
                    else:
                        temp=temp.replace('-08:00','')
                extend.append(temp)                           
                #if extend not in g1:
                #    g1.append(extend)
                if extend not in ex_triple:
                    ex_triple.append(extend)     
            # add ex_triple to g1
            random.shuffle(ex_triple)
            g1.extend(ex_triple)   
            random.shuffle(g1)
            ex_graph.append(g1)
            
        sample['qid']=one_example['qid']
        sample['question']=one_example['question']
        sample['answer']=one_example['answer']
        sample['sparql_query']=one_example['sparql_query']
        sample['s_expression']=one_example['s_expression']
        sample['graph']=graph
        sample['restrict_graph']=graph[:GRAPHNUM]
        sample['ex_graph']=ex_graph
        graphdata.append(sample)
            
    json.dump(graphdata,open('graph/'+file_type+'.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)

query('train')
query('dev')