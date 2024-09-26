# -*- coding: utf-8 -*-
# !/usr/bin/python

import pickle
import random
import json
import re
from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON

kb_endpoint = "http://10.201.173.146:3001//sparql"
'''
# load all the cvt relation
cvt_relation=set()
with open('cvt_relation.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip()
        cvt_relation.add(line)
'''
#@timeout(15)
def KB_query_with_timeout(_query, kb_endpoint):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    #sparql.setTimeout(5)
    response = sparql.query().convert()
    results = parse_query_results(response)
    return results

def query_rel(rel,kb_endpoint=kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x WHERE { ?head" + " ns:"+ rel + " ?x .}"
    results = KB_query(query,kb_endpoint)
    result=[]
    if len(results)>0:
        for i in results:
            result.append(i['x'].replace('http://rdf.freebase.com/ns/',''))
    return result

def query_relation(head, tail, kb_endpoint=kb_endpoint):
    results = []
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel WHERE { ns:"+ head + " ?rel ns:" + tail + " .}"
    query_results = KB_query(query, kb_endpoint)
    if len(query_results) > 0:
        results = [rel["rel"].replace("http://rdf.freebase.com/ns/","") for rel in query_results]
    '''
    query_re = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel WHERE { ns:"+ tail + " ?rel ns:" + head + " .}" 
    query_results_re = KB_query(query_re, kb_endpoint)
    if len(query_results_re) > 0:
        results_re = [rel["rel"].replace("http://rdf.freebase.com/ns/","") for rel in query_results_re]
    if len(results) > 1 or len(results_re) > 1 :
        print("relation nums between %s and %s GT 1." %(head, tail))
    '''
    return results

def KB_query(_query, kb_endpoint=kb_endpoint,max_retries=1000):
    """
    :param _query: sparql query statement
    :return:
    """
    retries=0
    while retries < max_retries:
        try:
            sparql = SPARQLWrapper(kb_endpoint)
            sparql.setQuery(_query)
            sparql.setReturnFormat(JSON)
            response = sparql.query().convert()
            results = parse_query_results(response)
            return results
        except Exception as e:
            retries += 1
    return None  # Return None or raise an exc

def query_ent_name(x,kb_endpoint=kb_endpoint):
    query = 'PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE {  ns:' + x + ' ns:type.object.name ?name . FILTER(LANG(?name) = "en")}'
    results = KB_query(query, kb_endpoint)
    if len(results) == 0:
        query = 'PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns:' + x + ' ns:common.topic.alias ?name . FILTER(LANG(?name) = "en")}'
        results = KB_query(query, kb_endpoint)
        if len(results) == 0:
            return None
    name = results[0]["name"]
    return name

def parse_query_results(response):
    if "boolean" in response:  # ASK
        results = [response["boolean"]]
    else:
        if len(response["results"]["bindings"]) > 0 and "callret-0" in response["results"]["bindings"][0]: # COUNT
            results = [int(response['results']['bindings'][0]['callret-0']['value'])]
        else:
            results = []
            for res in response['results']['bindings']:
                res = {k: v["value"] for k, v in res.items()}
                results.append(res)
    return results


def formalize(query):
    p_where = re.compile(r'[{](.*?)[}]', re.S)
    select_clause = query[:query.find("{")].strip(" ")
    select_clause = [x.strip(" ") for x in select_clause.split(" ")]
    select_clause = " ".join([x for x in select_clause if x != ""])
    select_clause = select_clause.replace("DISTINCT COUNT(?uri)", "COUNT(?uri)")

    where_clauses = re.findall(p_where, query)[0]
    where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
    triples = [[y.strip(" ") for y in x.strip(" ").split(" ") if y != ""]
               for x in where_clauses.split(". ")]
    triples = [" ".join(["?x" if y[0] == "?" and y[1] == "x" else y for y in x]) for x in triples]
    where_clause = " . ".join(triples)
    query = select_clause + "{ " + where_clause + " }"
    return query

def query_answers(query, kb_endpoint):
    query = formalize(query)
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()

    if "ASK" in query:
        results = [str(response["boolean"])]
    elif "COUNT" in query:
        tmp = response["results"]["bindings"]
        assert len(tmp) == 1 and ".1" in tmp[0]
        results = [tmp[0][".1"]["value"]]
    else:
        tmp = response["results"]["bindings"]
        results = [x["uri"]["value"] for x in tmp]
    return results

def query_des(x):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns:" + x + " ns:common.topic.description ?name .}"
    results = KB_query(query,"http://10.201.69.194:8890//sparql")
    '''
    if len(results) == 0:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns: " + x + " common.topic.notable_types ?name .}"
        results = KB_query(query, kb_endpoint)
        if len(results) == 0:
            print(x, "does not have name !")
            return x
    '''
    if len(results)>0:
        name = results[0]["name"]
    else:
        name=''
    return name

def query_type(x):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns:" + x + " ns:common.topic.notable_types ?name .}"
    results = KB_query(query,"http://10.201.69.194:8890//sparql")
    
    if len(results) == 0:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns: " + x + " ns:type.object.name ?name .}"
        results = KB_query(query, kb_endpoint)
    
    if len(results)>0:
        name = results[0]["name"]
        name=query_ent_name(name.split('/')[-1])
    else:
        name=''
    return name

def query_answer(x,y):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns:" + x + " ns:"+y+" ?name .}"
    results = KB_query(query,"http://10.201.69.194:8890//sparql")
    '''
    if len(results) == 0:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { " + x + " ns:common.topic.alias ?name .}"
        results = KB_query(query, kb_endpoint)
        if len(results) == 0:
            print(x, "does not have name !")
            return x
    '''
    temp=[]
    name=[]
    if len(results)>0:
        for i in range(0,len(results)):
            temp.append(results[i]["name"].split('/')[-1])
        for i in temp:
            name.append([i,query_ent_name(i)])
    else:
        name=[]
    return name
    
def query_en_relation(head, kb_endpoint = "http://10.201.69.194:8890//sparql"):
    results = []
    results_re = []
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel WHERE { ns:"+ head + " ?rel ?y .}"
    query_results = KB_query(query, kb_endpoint)
    for i in query_results:
        temp=i['rel'].replace("http://rdf.freebase.com/ns/","")
        if temp not in results:
            results.append(temp)
    return results
    '''
    if len(query_results) > 0:
        results = [rel["rel"].replace("http://rdf.freebase.com/ns/","") for rel in query_results]
    query_re = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel WHERE { ns:"+ tail + " ?rel ns:" + head + " .}" 
    query_results_re = KB_query(query_re, kb_endpoint)
    if len(query_results_re) > 0:
        results_re = [rel["rel"].replace("http://rdf.freebase.com/ns/","") for rel in query_results_re]
    if len(results) > 1 or len(results_re) > 1 :
        print("relation nums between %s and %s GT 1." %(head, tail))
        '''
    
def query_obj(head):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?value WHERE { ns:" + head + " ?rel ?value .}"
    results = KB_query(query,"http://10.201.69.194:8890//sparql")
    result=[]
    if len(results)>0:
        for i in results:
            if '/' not in i["value"]:
                result.append(i['value'])
    return result
    
def query_med(head):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?value WHERE { ns:" + head + " ns:medicine.drug_formulation.brand_names ?value .}"
    results = KB_query(query,"http://10.201.69.194:8890//sparql")
    result=[]
    if len(results)>0:
        for i in results:
            result.append(i['value'])
    return result

def query_en_triple(head, kb_endpoint = kb_endpoint):
    results = []
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel ?y WHERE { ns:"+ head + " ?rel ?y .}"
    query_results = KB_query(query, kb_endpoint)
    '''
    for i in query_results:
        if "http://rdf.freebase.com/ns/" not in i['y']:
            continue
        y=i['y'].replace("http://rdf.freebase.com/ns/","")
        if y[1]!='.':
            continue
        rel=i['rel'].replace("http://rdf.freebase.com/ns/","")
        temp=[head,rel,y]
        if temp not in results:
            results.append(temp)
    '''
    results=query_results
    return results
    
def query_2hop(head, kb_endpoint = kb_endpoint):
    results = set()
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel ?y WHERE { ns:"+ head + " ?rel ?y .}"
    query_results = KB_query(query, kb_endpoint)
    query=set()
    for i in query_results:
        if "http://rdf.freebase.com/ns/" not in i['y']:
            continue
        y=i['y'].replace("http://rdf.freebase.com/ns/","")
        if y[1]!='.':
            continue
        rel=i['rel'].replace("http://rdf.freebase.com/ns/","")
        temp=(head,rel,y)
        results.add(temp)
        query.add(y)
    for que in query:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel ?y WHERE { ns:"+ que + " ?rel ?y .}"
        query_results = KB_query(query, kb_endpoint)
        for i in query_results:
            if "http://rdf.freebase.com/ns/" not in i['y']:
                continue
            y=i['y'].replace("http://rdf.freebase.com/ns/","")
            if y[1]!='.':
                continue
            rel=i['rel'].replace("http://rdf.freebase.com/ns/","")
            temp=(que,rel,y)
            results.add(temp)
    return results
    
def query_sametype(head,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns:" + head + " ns:common.topic.notable_types ?name .}"
    results = KB_query(query, kb_endpoint)
    if len(results)!=0:
        name=results[0]['name'].replace("http://rdf.freebase.com/ns/","")
        entity=query_type(name)
        entity.discard(head)
        if len(entity)>0:
            entity=random.choice(list(entity))
            return entity
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns:" + head + " ns:type.object.type ?name .}"
    results = KB_query(query, kb_endpoint)
    name=''
    for i in results:
        if "http://rdf.freebase.com/ns/" not in i['name']:
            continue
        temp=i['name'].replace("http://rdf.freebase.com/ns/","")
        if 'base' in temp or 'common.topic' in temp:
            continue
        name=temp
        break
    # cannot find type
    if name=='' and len(results)!=0:
        name=results[-1]['name'].replace("http://rdf.freebase.com/ns/","")
    if name=='' and len(results)==0:
        return ''
    entity=query_type1(name)
    entity.discard(head)
    if len(entity)>0:
        entity=random.choice(list(entity))
        return entity
    else:
        return ''

# query type with common.topic.notable_types
def query_type(typelist,limit=100,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x WHERE { ?x ns:common.topic.notable_types ns:" + typelist + ".} LIMIT " + str(limit)
    results = KB_query(query, kb_endpoint)
    entity=set()
    for i in results:
        entity.add(i['x'].replace("http://rdf.freebase.com/ns/",""))
    return entity

# query type with type.object.type
def query_type1(typelist,limit=100,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x WHERE { ?x ns:type.object.type ns:" + typelist + ".} LIMIT " + str(limit)
    results = KB_query(query, kb_endpoint)
    entity=set()
    for i in results:
        entity.add(i['x'].replace("http://rdf.freebase.com/ns/",""))
    return entity
    
def query_entype(x,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ns:" + x + " ns:type.object.type ?name .}"
    results = KB_query(query, kb_endpoint)
    name=''
    for i in results:
        if "http://rdf.freebase.com/ns/" not in i['name']:
            continue
        temp=i['name'].replace("http://rdf.freebase.com/ns/","")
        if 'base' in temp or 'common.topic' in temp:
            continue
        name=temp
        break
    return name


def modify_relation_1hop(triple,limit=100,kb_endpoint = kb_endpoint):
    head=''
    relation=''
    tail=''
    entype=''
    if triple[0]=='x0':
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ?x ?y ns:" + triple[2] + ".} LIMIT " + str(limit)
        results = KB_query(query, kb_endpoint)
        if results is None:
            return [],[]
        for i in results:
            if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
                head=i['x'].replace('http://rdf.freebase.com/ns/','')
                relation=i['y'].replace('http://rdf.freebase.com/ns/','')
                if 'common' in relation or 'type' in relation: 
                    continue
                entype=query_entype(head)
                if entype!='' and relation!=triple[1] and head[1]=='.':
                    break
        # if no other relations or head is not entity
        if relation==triple[1] or 'common' in relation or 'type' in relation or head[1]!='.':
            return [],[]
        # if entype is not None
        if entype!='':
            return [head,relation,triple[2]],[head,'type.object.type',entype]
        # if entype is None
        return [head,relation,triple[2]],[head]
    if triple[2]=='x0':
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ns:" + triple[0] + " ?x ?y.} LIMIT " + str(limit)
        results = KB_query(query, kb_endpoint)
        if results is None:
            return [],[]
        for i in results:
            if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
                relation=i['x'].replace('http://rdf.freebase.com/ns/','')
                tail=i['y'].replace('http://rdf.freebase.com/ns/','')
                if 'common' in relation or 'type' in relation: 
                    continue
                entype=query_entype(tail)
                if entype!='' and relation!=triple[1] and tail[1]=='.':
                    break
        # if no other relations or tail is not entity
        if relation==triple[1] or 'common' in relation or 'type' in relation or tail[1]!='.':
            return [],[]
        # if entype is not None
        if entype!='':
            return [triple[0],relation,tail],[tail,'type.object.type',entype]
        # if entype is None
        return [triple[0],relation,tail],[tail]

'''
def modify_relation(triple,limit=100,kb_endpoint = kb_endpoint):
    head=''
    relation=''
    tail=''
    entype=''
    if triple[0]=='x0':
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ?x ?y ns:" + triple[2] + ".} LIMIT " + str(limit)
        results = KB_query(query, kb_endpoint)
        if results is None:
            return [],[]
        for i in results:
            if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
                head=i['x'].replace('http://rdf.freebase.com/ns/','')
                relation=i['y'].replace('http://rdf.freebase.com/ns/','')
                entype=query_entype(head)
                if entype!='' and relation!=triple[1] and head[1]=='.':
                    break
        # if no other relations or head is not entity
        if relation==triple[1] or head[1]!='.':
            return [],[]
        # if entype is not None
        if entype!='':
            return [head,relation,triple[2]],[head,'type.object.type',entype]
        # if entype is None
        return [head,relation,triple[2]],[head]
    if triple[2]=='x0':
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ns:" + triple[0] + " ?x ?y.} LIMIT " + str(limit)
        results = KB_query(query, kb_endpoint)
        if results is None:
            return [],[]
        for i in results:
            if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
                relation=i['x'].replace('http://rdf.freebase.com/ns/','')
                tail=i['y'].replace('http://rdf.freebase.com/ns/','')
                entype=query_entype(tail)
                if entype!='' and relation!=triple[1] and tail[1]=='.':
                    break
        # if no other relations or tail is not entity
        if relation==triple[1] or tail[1]!='.':
            return [],[]
        # if entype is not None
        if entype!='':
            return [triple[0],relation,tail],[tail,'type.object.type',entype]
        # if entype is None
        return [triple[0],relation,tail],[tail]
'''    
        
def generate_head(x,rel,limit=1000,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ?x ?y ns:" + x + ".} LIMIT " + str(limit)
    results = KB_query(query, kb_endpoint)
    for i in results:
        if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
            head=i['x'].replace('http://rdf.freebase.com/ns/','')
            relation=i['y'].replace('http://rdf.freebase.com/ns/','')
            if 'common' in relation or 'type' in relation: 
                continue
            if relation==rel:
                continue
            return [head,relation,x]

def generate_tail(x,rel,limit=1000,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ns:" + x + " ?x ?y.} LIMIT " + str(limit)
    results = KB_query(query, kb_endpoint)
    for i in results:
        if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
            relation=i['x'].replace('http://rdf.freebase.com/ns/','')
            tail=i['y'].replace('http://rdf.freebase.com/ns/','')
            if 'common' in relation or 'type' in relation: 
                continue
            if relation==rel:
                continue
            return [x,relation,tail]

# generate triple without relation restrict     
def generate_head1(x,limit=1000,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ?x ?y ns:" + x + ".} LIMIT " + str(limit)
    results = KB_query(query, kb_endpoint)
    for i in results:
        if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
            head=i['x'].replace('http://rdf.freebase.com/ns/','')
            relation=i['y'].replace('http://rdf.freebase.com/ns/','')
            if 'common' in relation or 'type' in relation: 
                continue
            return [head,relation,x]

# generate triple without relation restrict 
def generate_tail1(x,limit=1000,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x ?y WHERE { ns:" + x + " ?x ?y.} LIMIT " + str(limit)
    results = KB_query(query, kb_endpoint)
    for i in results:
        if 'http://rdf.freebase.com/ns/' in i['x'] and 'http://rdf.freebase.com/ns/' in i['y']:
            relation=i['x'].replace('http://rdf.freebase.com/ns/','')
            tail=i['y'].replace('http://rdf.freebase.com/ns/','')
            if 'common' in relation or 'type' in relation: 
                continue
            return [x,relation,tail]
            
# get 1 hop triple
def get_1hop(head,kb_endpoint = kb_endpoint):
    skip=['type.object.name','type.object.type','common.topic.description','common.topic.notable_for','common.topic.article','common.topic.topic_equivalent_webpage','common.topic.alias',"common.topic.webpage",'common.topic.topical_webpage',"common.topic.notable_types","type.object.key","kg.object_profile.prominent_type"]
    results = []
    #query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel ?y WHERE { ns:"+ head + " ?rel ?y . FILTER (!regex(str(?rel), "^ns:common.topic")) FILTER (?rel NOT IN (ns:type.object.name, ns:type.object.type, ns:common.topic.description, ns:common.topic.notable_for, ns:common.topic.article, ns:common.topic.topic_equivalent_webpage,ns:common.topic.alias,ns:common.topic.webpage,ns:common.topic.topical_webpage,ns:common.topic.notable_types,ns:type.object.key,ns:kg.object_profile.prominent_type,ns:common.topic.official_website,ns:common.topic.image,ns:common.topic.official_website))} LIMIT 50"
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel ?y WHERE { ns:"+ head + " ?rel ?y . FILTER (!STRSTARTS(STR(?rel), 'http://rdf.freebase.com/ns/common.topic')) FILTER (!STRSTARTS(STR(?rel), 'http://rdf.freebase.com/ns/type.object')) FILTER (!STRSTARTS(STR(?rel), 'http://rdf.freebase.com/key')) FILTER (!STRENDS(STR(?rel), 'type')) FILTER (!STRENDS(STR(?rel), 'label'))} LIMIT 50"
    query_results = KB_query(query, kb_endpoint)
    for i in query_results:
        if "http://rdf.freebase.com/ns/" not in i['rel']:
            continue
        rel=i['rel'].replace("http://rdf.freebase.com/ns/","")
        if rel in skip:
            continue
        if "http://rdf.freebase.com/ns/" not in i['y']:
            y=i['y']
        else:
            y=i['y'].replace("http://rdf.freebase.com/ns/","")
        temp=[head,rel,y]
        if temp not in results:
            results.append(temp)
    return results
    
# modify cvt node
def generate_cvt(head,relation,DIR,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?rel ?y WHERE { ns:"+ head + " ?rel ?y .}"
    query_results = KB_query(query, kb_endpoint)
    for i in query_results:
        if "http://rdf.freebase.com/ns/" not in i['rel'] or "http://rdf.freebase.com/ns/" not in i['y']:
            continue
        rel=i['rel'].replace("http://rdf.freebase.com/ns/","")
        y=i['y'].replace("http://rdf.freebase.com/ns/","")
        if rel in cvt_relation and rel!=relation:
            if DIR=='LEFT':
                new_tri=generate_head1(y)
            if DIR=='RIGHT':
                new_tri=generate_tail1(y)
            if new_tri is not None:
                return [head,rel,y],new_tri
    return None, None

# query entity with the same name
def query_same_name(mid,name,kb_endpoint = kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x WHERE { ?x ns:type.object.name \""+ name+"\"@en .}"
    results = KB_query(query, kb_endpoint)
    if not results:
        return None
    if len(results) <2:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { ?x ns:common.topic.alias '"+ name+"'@en .}"
        results = KB_query(query, kb_endpoint)
        if not results:
            return None
        if len(results) < 2:
            return None
    entity=set()
    for i in results:
        entity.add(i["x"].replace("http://rdf.freebase.com/ns/",""))
    entity.discard(mid)
    entity=list(entity)
    return entity[:2]
    
if __name__ == '__main__':
    #print(query_type('architecture.architect'))
    #print(query_sametype('g.120w89tg'))
    #print(generate_tail1('m.0_qxwd5'))
    #print(query_entype('m.0ttqrhs'))
    #print(get_1hop('m.0jvd79q'))
    print(query_ent_name('m.0jvd79q'))
    #print(query_same_name('m.03_r3','Jamaica'))
    #print(query_relation("m.0ddt_","m.0k3qy8"))
    