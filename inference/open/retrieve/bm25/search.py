# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from pyserini.search.lucene import LuceneSearcher
import json
from tqdm import tqdm
import os
import re
import argparse
import pickle
import multiprocessing.pool
from functools import partial
from collections import defaultdict
from pyserini.index import IndexReader
        
def has_digit(input_string):
    for char in input_string:
        if char.isdigit():
            return True
    return False

class Bm25Searcher:
    def __init__(self, index_dir, args):
        self.index_dir = index_dir
        self.args = args
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(args.k1, args.b)
        self.index_reader=IndexReader(index_dir)
        if len(args.ignore_string) > 0:
            self.ignore_list = args.ignore_string.split(',')
            print(f'ignore list: {self.ignore_list}')
        else:
            self.ignore_list = []
        
        # load documents for post process
        '''
        self.mid2doc=dict()
        for path in tqdm(os.listdir(self.args.documents)):
            with open(self.args.documents+'/'+path,'r',encoding='utf-8') as f:
                for line in f:
                    try:
                        data=json.loads(line)
                        if self.mid2doc.get(data["mid"]) is None:
                            self.mid2doc[data["mid"]]=[]
                        self.mid2doc[data["mid"]].append(data["id"])
                    except:
                        continue
        with open('mid2doc.pickle', 'wb') as f:
            pickle.dump(self.mid2doc, f)
        '''
        with open('mid2doc.pickle', 'rb') as f:
            self.mid2doc = pickle.load(f)   
 
    def perform_search(self, sample, top_k, ques_id):
        if self.args.dataset in ['WebQSP']:
            query=sample["question"]
            head=set()
            head.add(sample["headmid"])
        if self.args.dataset in ['GraphQuestions','grailqa']:
            query=sample["question"]
            head=set()
            for n in sample["graph_query"]["nodes"]:
                if n["node_type"]=="entity":
                    head.add(n["id"])
        for string in self.ignore_list:
            query = query.replace(string, ' ')
        query = query.strip()
        
        # get relevant document for head entity
        docid=[]
        for i in head:
            if self.mid2doc.get(i):
                docid.extend(self.mid2doc[i])
        # first search using relevant document
        id_score=[]
        for i in docid:
            score = self.index_reader.compute_query_document_score(i, query)
            id_score.append([score,i])
        id_score=sorted(id_score,key=lambda x: x[0], reverse=True)
        documents=[]
        for i in id_score[:top_k]:
            raw_data=self.searcher.doc(i[1])
            documents.append(json.loads(raw_data.raw()))

        # search
        if len(documents)<top_k:
            results = self.searcher.search(query, k=200)
            for result in results:
                try:
                    doc_dict = json.loads(result.lucene_document.get('raw'))
                    if doc_dict not in documents:
                        documents.append(doc_dict)
                        if len(documents)>=top_k:
                            break
                except:
                    continue
        context = dict()
        context['documents']=documents
        context['id']=ques_id
        return context

def search_all(process_idx, num_process, searcher, args):
    # load dataset
    with open(args.query_data_path, 'r') as rf:
        data = json.load(rf)

    output_data = []
    for i, data_i in tqdm(enumerate(data)):
        if i % num_process != process_idx:
            continue
        # search
        output_i = searcher.perform_search(data_i, args.top_k,i)
        output_data.append(output_i)
    return output_data
    
def eval_top_k_one(documents, answer,top_k):
    recall = 0
    # merge into context
    context=''
    for doc in documents['documents'][:top_k]:
        context+=doc['triples']
    for ans in answer:
        if ans.lower() in context.lower():
            recall += 1
    return recall / (len(answer) + 1e-8)    

def eval_top_k(output_data, answers,top_k_list=[1,2,3,4,5,6,7,8,9,10]):
    print("*"*30,"Evaluate the Retrieval Result","*"*30)
    hits_dict = defaultdict(int)
    recall_dict = defaultdict(float)
    top_k_list = [k for k in top_k_list if k <= len(output_data[0]['documents'])]
    for documents,answer in tqdm(zip(output_data,answers)):
        for k in top_k_list:
            recall = eval_top_k_one(documents, answer,k)
            if recall > 0:
                hits_dict[k] += 1
            recall_dict[k] += recall
    for k in top_k_list:
        print("Top {}".format(k), 
              "Hits: ", round(hits_dict[k] * 100 / len(output_data), 1), 
              "Recall: ", round(recall_dict[k] * 100 / len(output_data), 1))

# argparse for root_dir, index_dir, query_data_path, output_dir
parser = argparse.ArgumentParser(description='Search using pySerini')
parser.add_argument("--dataset", type=str, default='WebQSP',
                    help="KBQA dataset")
parser.add_argument("--documents", type=str, default='../../Freebase/processed/document',
                    help="documents directory")                    
parser.add_argument("--index_name", type=str, default='Wikidata',
                    help="directory to store the search index")
parser.add_argument("--query_data_path", type=str, default='',
                    help="directory to store the queries")
parser.add_argument("--output", type=str, default='',
                    help="directory to store the retrieved output")
parser.add_argument("--num_process", type=int, default=10,
                    help="number of processes to use for multi-threading")
parser.add_argument("--top_k", type=int, default=150,
                    help="number of passages to be retrieved for each query")
parser.add_argument("--ignore_string", type=str, default="",
                    help="string to ignore in the query, split by comma")
parser.add_argument("--b", type=float, default=0.4,
                    help="parameter of BM25")
parser.add_argument("--k1", type=float, default=0.9,
                    help="parameter of BM25")
parser.add_argument("--save", action="store_true",
                    help="whether to save the output")
parser.add_argument("--eval", action="store_true",
                    help="whether to evaluate the output")
args = parser.parse_args()


if __name__ == '__main__':
    index_dir = args.index_name
    searcher = Bm25Searcher(index_dir, args)

    num_process = args.num_process
    pool = multiprocessing.pool.ThreadPool(processes=num_process)
    sampleData = [x for x in range(num_process)]
    search_all_part = partial(search_all, 
                                searcher = searcher,
                                num_process = num_process,
                                args = args)
    results = pool.map(search_all_part, sampleData)
    pool.close()

    output_data = []
    for result in results:
        output_data += result

    # sort the output data by question id
    output_data = sorted(output_data, key=lambda item: item['id'])
    if args.eval:
        # load answer from original data
        answers=[]
        with open(args.query_data_path, 'r') as rf:
            dataset = json.load(rf)
        if args.dataset in ['WebQSP']:
            for sample in dataset:  
                answers.append(sample["answername"].split('|'))
        if args.dataset in ['GraphQuestions','grailqa']:
            for sample in dataset:
                answer=[]  
                for i in sample["answer"]:   
                    if i.get("entity_name"):
                        answer.append(i["entity_name"])
                    else:
                        answer.append(i["answer_argument"]) 
                answers.append(answer)
        # evaluate output                   
        eval_top_k(output_data, answers, top_k_list=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
    
    # truncate documents into 10 documents
    for i in output_data:
        i['documents'] = i['documents'][:10]

    # save output data
    # create output dir recursively if not exist
    if args.save:
        os.makedirs('results', exist_ok=True)
        print("saving output data to {}".format(args.output))
        with open(args.output, "w") as wf:
            json.dump(output_data, wf, indent=2, ensure_ascii=False)
