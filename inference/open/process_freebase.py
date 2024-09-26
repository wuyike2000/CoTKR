# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import jsonlines
import csv
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
import argparse
from linearize import Relation, convert_relation_to_text
from query_interface import query_ent_name

# DATA_DIR = os.environ['DATA_DIR']
TRIPLE_NUM=10

parser = argparse.ArgumentParser(description='process Freebase')
parser.add_argument('--data_dir', type=str, default='Freebase')
args = parser.parse_args()

'''
# disambuity for entity name
# if the entity name has appeared in previous ids, we add indicator like "v1" "v2" to the name
name_dir = os.path.join(args.data_dir, "id2name_parts")
output_dir = os.path.join(args.data_dir, "id2name_parts_disamb")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    name_num_dict = defaultdict(int)
    file_list = os.listdir(name_dir)
    file_list.sort()
    for file_name in tqdm(file_list):
        print(file_name)
        id_name_list = []
        with open(os.path.join(name_dir, file_name), 'r') as rf:
            data_input = csv.reader(rf, delimiter="\t")
            for row in data_input:
                if row[2] not in name_num_dict:
                    id_name_list.append(row)
                    name_num_dict[row[2]] += 1
                else:
                    new_row = row[:2] + [row[2] + " v" + str(name_num_dict[row[2]])]
                    id_name_list.append(new_row)
                    name_num_dict[row[2]] += 1
        # save the list of rows to a new tsv file
        with open(os.path.join(output_dir, file_name), 'w') as wf:    
            data_output = csv.writer(wf, delimiter="\t")
            for row in id_name_list:
                data_output.writerow(row)


# load Freebase entity name
name_dir = os.path.join(args.data_dir, "id2name_parts_disamb")
id2name_dict = {}
for file_name in tqdm(os.listdir(name_dir)):
    with open(os.path.join(name_dir, file_name), 'r') as rf:
        data_input = csv.reader(rf, delimiter="\t")
        for row in data_input:
            id2name_dict[row[0]] = row[2]
print("number of entities with names: ", len(id2name_dict))
'''

# load Freebase entity name
name_dir = os.path.join(args.data_dir, "id2name_parts")
id2name_dict = {}
for file_name in tqdm(os.listdir(name_dir)):
    with open(os.path.join(name_dir, file_name), 'r') as rf:
        data_input = csv.reader(rf, delimiter="\t")
        for row in data_input:
            id2name_dict[row[0]] = row[2]
print("number of entities with names: ", len(id2name_dict))

# load topic and type information from Freebase
topic_dir = os.path.join(args.data_dir, "topic_entities_parts")
id2topic_dict = defaultdict(set)
file_list = os.listdir(topic_dir)
# sort file list
file_list.sort()
for file_name in tqdm(file_list):
    with open(os.path.join(topic_dir, file_name), 'r') as rf:
        for row in rf:
            row = row.strip().split("\t")
            if len(row) != 3:
                continue
            if "web" in row[1] or "type.object.name" in row[1]:
                continue
            # only consider the last part of relation
            topic = row[1].split(".")[-1].replace('.', ' ').replace('_', ' ')
            topic += " : " + row[2].replace('.', ' ').replace('_', ' ')
            id2topic_dict[row[0]].add(topic)
print("number of entities with topics: ", len(id2topic_dict))

'''
# transform each entity centric 1-hop subgraph (only out relations) as one passage
def transform_triples_group_woid(process_idx, num_process, triple_dir, save_dir):
    # do not keep CVT node id
    for file_name in tqdm(os.listdir(triple_dir)):
        file_id = int(file_name.split("-")[-1])
        if file_id % num_process != process_idx:
            continue
        file_path = os.path.join(triple_dir, file_name)
        grouped_entity_triples = defaultdict(list)
        with open(file_path, "r") as rf:
            for row in rf:
                if len(row.strip().split("\t")) != 3:
                    continue
                triple = Relation(row)
                if triple.should_ignore(id2name_dict):
                    continue
                #if triple.obj.startswith("m") and triple.obj not in id2name_dict:
                #    continue
                #subj = triple.subj
                #if triple.subj not in id2name_dict:
                #    triple.subj = ""
                grouped_entity_triples[subj].append(convert_relation_to_text(triple, id2name_dict))
                
        with jsonlines.open(os.path.join(save_dir, file_name) + ".jsonl", "w") as wf:
            data_id = 0
            for key in grouped_entity_triples:
                if key in id2name_dict:
                    name_key = id2name_dict[key]
                    title = name_key + " " + key
                else:
                    name_key = ""
                    title = key
                contents = " ".join(grouped_entity_triples[key])
                # add topic information
                if key in id2topic_dict:
                    topics = id2topic_dict[key]
                    topics = [name_key + " " + topic + " ." for topic in topics]
                    topics_contents = " ".join(topics)
                    contents += " " + topics_contents
                
                contents = contents.replace("\t", " ").replace("\n", " ")
                title = title.replace("\t", " ").replace("\n", " ")
                # split contents by 100 words each chunk
                contents_split = contents.split(" ")
                contents_chunks = [" ".join(contents_split[i:i+100]) for i in range(0, len(contents_split), 100)]
                for contents_chunk in contents_chunks:
                    data_output_i = {"id": "freebase-file{}doc{}".format(file_id, data_id),
                                    "contents": contents_chunk,
                                    "title": title}
                    wf.write(data_output_i)
                    data_id += 1
'''

# transform each entity centric 1-hop subgraph (only out relations) as one passage
def transform_triples_group_woid(process_idx, num_process, triple_dir, save_dir):
    # do not keep CVT node id
    for file_name in tqdm(os.listdir(triple_dir)):
        file_id = int(file_name.split("-")[-1])
        if file_id % num_process != process_idx:
            continue
        file_path = os.path.join(triple_dir, file_name)
        grouped_entity_triples = defaultdict(dict)
        with open(file_path, "r") as rf:
            for row in rf:
                if len(row.strip().split("\t")) != 3:
                    continue
                triple = Relation(row)
                if triple.should_ignore(id2name_dict):
                    continue
                
                # if triple.obj.startswith("m") and triple.obj not in id2name_dict:
                #     continue
                # subj = triple.subj
                # if triple.subj not in id2name_dict:
                #     triple.subj = ""
                
                triple_form,text_form=convert_relation_to_text(triple, id2name_dict)
                
                if grouped_entity_triples.get(row.strip().split("\t")[0]) is None:
                    grouped_entity_triples[row.strip().split("\t")[0]]={'text':[],'triple':[]}
                
                grouped_entity_triples[row.strip().split("\t")[0]]['text'].append(text_form)
                grouped_entity_triples[row.strip().split("\t")[0]]['triple'].append(triple_form)
                
        with jsonlines.open(os.path.join(save_dir, file_name) + ".jsonl", "w") as wf:
            data_id = 0
            for key in grouped_entity_triples:
                if key in id2name_dict:
                    key_title = id2name_dict[key]
                else:
                    key_title = ""
                key_mid=key
                
                # topic information
                if key in id2topic_dict:
                    topics = id2topic_dict[key]
                    if len(key_title)!=0:
                        topics = [key_title + " " + topic + " ." for topic in topics]
                    else:
                        topics = [topic + " ." for topic in topics]
                    topics_contents = " ".join(topics)
                
                # split contents by TRIPLE_NUM triples each chunk
                for i in range(0,len(grouped_entity_triples[key]['text']),TRIPLE_NUM):
                    contents=" ".join(grouped_entity_triples[key]['text'][i:i+TRIPLE_NUM])
                    triples=" ".join(grouped_entity_triples[key]['triple'][i:i+TRIPLE_NUM])
                    if len(topics_contents)!=0:
                        contents=contents+' '+topics_contents
                    contents = contents.replace("\t", " ").replace("\n", " ")
                    triples = triples.replace("\t", " ").replace("\n", " ")
                    key_title = key_title.replace("\t", " ").replace("\n", " ")
                    data_output_i = {"id": "freebase-file{}doc{}".format(file_id, data_id),
                                    "contents": contents,
                                    "triples": triples,
                                    "name": key_title,
                                    "mid": key_mid}
                    wf.write(data_output_i)
                    data_id += 1

'''                
# transform each entity centric 1-hop subgraph (only out relations) as one passage
def transform_triples(triple_dir, save_dir):
    # do not keep CVT node id
    for file_name in tqdm(os.listdir(triple_dir)):
        file_id = int(file_name.split("-")[-1])
        file_path = os.path.join(triple_dir, file_name)
        grouped_entity_triples = defaultdict(dict)
        with open(file_path, "r") as rf:
            for row in rf:
                if len(row.strip().split("\t")) != 3:
                    continue
                triple = Relation(row)
                if triple.should_ignore(id2name_dict):
                    continue
                
                triple_form,text_form=convert_relation_to_text(triple, id2name_dict)
                
                if grouped_entity_triples.get(row.strip().split("\t")[0]) is None:
                    grouped_entity_triples[row.strip().split("\t")[0]]={'text':[],'triple':[]}
                
                grouped_entity_triples[row.strip().split("\t")[0]]['text'].append(text_form)
                grouped_entity_triples[row.strip().split("\t")[0]]['triple'].append(triple_form)
                
        with jsonlines.open(os.path.join(save_dir, file_name) + ".jsonl", "w") as wf:
            data_id = 0
            for key in grouped_entity_triples:
                # check whether it is literal
                # literal
                if key[:2] not in ['m.','n.','g.']:
                    key_title = key.replace('-08:00','')
                # entity
                else:
                    if key in entity_names:
                        key_title = entity_names[key]
                    else:
                        temp = query_ent_name(obj)
                        if not temp:
                            key_title = ""
                        else:
                            key_title = temp
                key_mid=key
                
                # topic information
                if key in id2topic_dict:
                    topics = id2topic_dict[key]
                    if len(key_title)!=0:
                        topics = [key_title + " " + topic + " ." for topic in topics]
                    else:
                        topics = [topic + " ." for topic in topics]
                    topics_contents = " ".join(topics)
                
                # split contents by TRIPLE_NUM triples each chunk
                for i in range(0,len(grouped_entity_triples[key]['text']),TRIPLE_NUM):
                    contents=" ".join(grouped_entity_triples[key]['text'][i:i+TRIPLE_NUM])
                    triples=" ".join(grouped_entity_triples[key]['triple'][i:i+TRIPLE_NUM])
                    if len(topics_contents)!=0:
                        contents=contents+' '+topics_contents
                    contents = contents.replace("\t", " ").replace("\n", " ")
                    triples = triples.replace("\t", " ").replace("\n", " ")
                    key_title = key_title.replace("\t", " ").replace("\n", " ")
                    data_output_i = {"id": "freebase-file{}doc{}".format(file_id, data_id),
                                    "contents": contents,
                                    "triples": triples,
                                    "name": key_title,
                                    "mid": key_mid}
                    wf.write(data_output_i)
                    data_id += 1
'''

# Multi-process data processing
def transform_triples_multip(triple_dir, save_dir):
    num_process = 10
    pool = Pool(num_process)
    sampleData = [x for x in range(num_process)]
    transform_triples_part = partial(transform_triples_group_woid, 
                                     triple_dir=triple_dir, 
                                     num_process=num_process, 
                                     save_dir=save_dir)
    pool.map(transform_triples_part, sampleData)
    pool.close()
    pool.join()

triple_dir = os.path.join(args.data_dir, "triple_edges_parts")
save_dir = os.path.join(args.data_dir, "processed/document")
os.makedirs(save_dir, exist_ok=True)
transform_triples_multip(triple_dir, save_dir)
