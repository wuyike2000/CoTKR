import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import math
import time
from tqdm import tqdm
import copy
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from openai import OpenAI
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

generation_config = GenerationConfig(
        temperature=1,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        num_return_sequences=6,
        max_new_tokens=1024
    )
    
# generation config
generation_config1 = GenerationConfig(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=1024
)

# resume
resume=0
# dataset
DATA='grailqa'
# llm: Meta-Llama-3-8B-Instruct, llama-2-7b-chat-hf
LLM='llama-2-7b-chat-hf'
# mode: CoT, kg-to-text
MODE='CoT'
# set EX_RATE
if DATA in ['GraphQuestions','WebQSP']:
    EX_RATE=1
if DATA in ['grailqa']:
    EX_RATE=1
# set save interval
INTERVAL=500
# set length threshold, avoid too short answers
LENGTH=20
# set similarity threshhold: avoid too similar knowledge representation
SIM_RATE=0.9
# set min number of triples
MIN_NUM=30
# set max length
MAX_LENGTH=3000
# set client
client=OpenAI(api_key='YOUR KEY')

# path for LLM
LLM_PATH='../../pretrain/'+LLM
# path for tokenizer
TOKENIZER_PATH='../../pretrain/'+LLM
# path for lora
PEFT_PATH='../instruction-tuning/output-'+DATA+'/'+MODE+'/'+LLM+'/best_model'
# result for checking
result=DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/sample.json'
# DPO data construction
DPO_data=DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/DPO.json'
DPO_train=DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/train.json'
DPO_dev=DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/dev.json'
os.makedirs(DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/middle',exist_ok = True)
# set similarity embedding model
sim_model=SentenceTransformer('../../pretrain/all-MiniLM-L6-v2',device='cpu')

# evaluate prompt
eval_prompt='''Your task is to evaluate the quality of two responses to the question based on predefined criteria. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
For this evaluation, you should primarily consider the following criteria:
Accuracy: The response should contain as many answer entities as possible, and use the original words of the answer entities.
Relevance: The response should be to the point of the question.
Question: {question}
Answer: {answer}
Response A: {response1}
Response B: {response2}
Begin your evaluation by comparing the two responses and provide a short explanation. Then output only the single character: "A" if Response A is better, "B" if Response B is better, and "C" for a tie. At the end, repeat just the letter again by itself on a new line.'''

# kr prompt
if MODE=='CoT':
    kr_prompt='''Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: {triple}
Question: {ques}
'''

if MODE=='kg-to-text':
    kr_prompt='Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {triple}. The sentence is:'

# ans prompt
ans_prompt='''Your task is to answer the question based on the knowledge that might be relevant. Try to use the original words from the given knowledge to answer the question. But if it is not useful, just ignore it and generate your own guess.
{knowledge}
Question: {ques}
Answer: '''

# paraphrase prompt
para_prompt='''You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Triple", "Answer" and "Knowledge". Your task is to paraphrase the original "Knowledge" into a more helpful representation format for Question Answering. The "Paraphrased Knowledge" should contain the original words of all the answer entities.
Question: bauman moscow state technical university can provide the what type of education?
Triple: (Oleg Skripochka, people.person.education, CVT3) (CVT3, education.education.institution, Bauman Moscow State Technical University) (Oleg Grigoryevich Makarov, people.person.education, CVT4) (CVT4, education.education.institution, Bauman Moscow State Technical University) (Konstantin Feoktistov, people.person.education, CVT1) (CVT1, education.education.institution, Bauman Moscow State Technical University) (CVT2, education.education.institution, Bauman Moscow State Technical University) (Yelena Kondakova, people.person.education, CVT2)
Answer: Konstantin Feoktistov|Yelena Kondakova|Oleg Skripochka|Oleg Grigoryevich Makarov
Knowledge: 
Reason 1: I need to know what type of education Bauman Moscow State Technical University provides.
Knowledge 1: Bauman Moscow State Technical University is associated with providing education for individuals such as Oleg Skripochka, Oleg Grigoryevich Makarov, Konstantin Feoktistov, and Yelena Kondakova.
Reason 2: I need to determine the specific type of education provided by Bauman Moscow State Technical University based on the individuals mentioned.
Knowledge 2: The type of education provided by Bauman Moscow State Technical University includes education in fields related to technology and engineering.
Paraphrased Knowledge:
Reason 1: I need to know what type of education Bauman Moscow State Technical University provides.
Knowledge 1: Bauman Moscow State Technical University is associated with providing education for individuals such as Oleg Skripochka, Oleg Grigoryevich Makarov, Konstantin Feoktistov, and Yelena Kondakova.

You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Triple", "Answer" and "Knowledge". Your task is to paraphrase the original "Knowledge" into a more helpful representation format for Question Answering. The "Paraphrased Knowledge" should contain the original words of all the answer entities.
Question: how does the drug tramadol take effect?
Triple: (Tramadol, medicine.drug.mechanism_of_action, Full Opioid Agonist)
Answer: Full Opioid Agonist
Knowledge: 
Reason 1: I need to know the mechanism of action of the drug Ultram (Tramadol).
Knowledge 1: Tramadol is a full opioid agonist, which means it works by binding to opioid receptors in the brain and spinal cord to produce pain relief.
Paraphrased Knowledge:
Reason 1: I need to know the mechanism of action of the drug Ultram (Tramadol).
Knowledge 1: Tramadol acts as a full opioid agonist through its mechanism of action.

You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Triple", "Answer" and "Knowledge". Your task is to paraphrase the original "Knowledge" into a more helpful representation format for Question Answering. The "Paraphrased Knowledge" should contain the original words of all the answer entities.
Question: which martial art category is t'ai chi ch'uan in?
Triple: (Internal, martial_arts.martial_art_category.martial_arts, Tai chi) (Grappling, martial_arts.martial_art_category.martial_arts, Tai chi) (Strike, martial_arts.martial_art_category.martial_arts, Tai chi)
Answer: Strike|Grappling|Internal
Knowledge: 
Reason 1: I need to know the category of martial art that T'ai Chi Ch'uan belongs to.
Knowledge 1: T'ai Chi Ch'uan is categorized as a type of martial art within the category of Tai chi.
Paraphrased Knowledge:
Reason 1: I need to know the category of martial art that T'ai Chi Ch'uan belongs to.
Knowledge 1: T'ai chi ch'uan is categorized as an Internal, Grappling and Strike martial art.

You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Triple", "Answer" and "Knowledge". Your task is to paraphrase the original "Knowledge" into a more helpful representation format for Question Answering. The "Paraphrased Knowledge" should contain the original words of all the answer entities.
Question: {question}
Triple: {triple}
Answer: {answer}
Knowledge: 
{knowledge}
Paraphrased Knowledge:
'''

num_dict = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
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
            
def LLMResponse(prompt,llm,tokenizer,cuda):
    inputs = tokenizer(prompt,return_tensors="pt")
    generation_output = llm.generate(
            input_ids=inputs["input_ids"].to(cuda),
            attention_mask=inputs['attention_mask'].to(cuda),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config
        )
    response=set()
    for i in generation_output:
        output = tokenizer.decode(i,skip_special_tokens=True)
        response.add(output.split(prompt)[-1].strip())
    return response
    
def LLMSingleResponse(prompt,llm,tokenizer,cuda):
    inputs = tokenizer(prompt,return_tensors="pt")
    generation_output = llm.generate(
            input_ids=inputs["input_ids"].to(cuda),
            attention_mask=inputs['attention_mask'].to(cuda),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config1
        )
    output = tokenizer.decode(generation_output[0],skip_special_tokens=True)
    response = output.split(prompt)[-1].strip()
    return response

def sim_compute(sentence):
    embeddings = sim_model.encode(sentence, batch_size=16,device='cpu',convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    pairs = []
    for i in range(cosine_scores.shape[0]):
        for j in range(i+1,cosine_scores.shape[1]):
            pairs.append({"index": [i, j], "score": cosine_scores[i][j]})
    pairs = sorted(pairs, key=lambda x: x["score"], reverse=False)
    i, j = pairs[0]["index"]
    return [sentence[i], sentence[j]]

# load toknizer and llm
tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH)
llm=AutoModelForCausalLM.from_pretrained(LLM_PATH,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='cuda:0')

# merge peft into base LLM
if PEFT_PATH:
    llm=PeftModel.from_pretrained(llm, PEFT_PATH,torch_dtype=torch.float16,device_map='cuda:0')
llm.cuda()
llm.eval()

# load data
data=json.load(open('../subgraph/'+DATA+'/graph/train.json','r',encoding='utf-8'))

# record
record=[]
# DPO training data
DPO=[]
# index
index=0
if resume>0:
    record=json.load(open(DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/middle/sample-'+str(resume)+'.json','r',encoding='utf-8'))
    DPO=json.load(open(DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/middle/DPO-'+str(resume)+'.json','r',encoding='utf-8'))
    index=resume

with torch.no_grad():
    for sample in tqdm(data[resume:]):
        # save intervally
        if index%INTERVAL==0:
            # path for interval result
            result_interval=DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/middle/sample-'+str(index)+'.json'
            # path for interval dpo data
            DPO_data_interval=DATA+'/PA-chatgpt/'+MODE+'/'+LLM+'/middle/DPO-'+str(index)+'.json'
            # save
            json.dump(DPO,open(DPO_data_interval,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
            json.dump(record,open(result_interval,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
        # add index
        index+=1
        # graph sample
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
        NUM=max(math.ceil(len(gold_g)*EX_RATE),MIN_NUM)
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
        
        # gold graph
        gold_string=' '.join(gold_g)
        
        # noisy graph
        noisy=set(gold_g).union(ex_filter)
        # random shuffle
        noisy=list(noisy)
        random.shuffle(noisy)
        # noisy graph string
        noisy_string=''
        for i in noisy:
            noisy_string=noisy_string+i+' '
        noisy_string=noisy_string.strip()
        # prompt construction
        if MODE=='CoT':
            prompt=kr_prompt.format(triple=noisy_string,ques=sample["question"])
        if MODE=='kg-to-text':
            prompt=kr_prompt.format(triple=noisy_string)
        print(prompt)
        # avoid too long input which may cause OOM
        if len(tokenizer(prompt,return_tensors="pt")["input_ids"][0])>MAX_LENGTH:
            continue
        # sample candidate knowledge representation format
        knowledge=LLMResponse(prompt,llm,tokenizer,'cuda:0')
        knowledge = [i for i in list(knowledge) if 'Reason' in i and 'Knowledge' in i and len(i) > LENGTH]
      
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

        # whether use chatgpt to paraphrase the knowledge
        # select different knowledge representation based on semantic similarity
        if len(knowledge)==0:
            continue
        # if knowledge is one, use chatgpt to paraphrase
        if len(knowledge)==1:
            para_know=getResponse(para_prompt.format(question=sample["question"],triple=gold_string,answer='|'.join(gold),knowledge=knowledge[0]))
            print(para_prompt.format(question=sample["question"],triple=gold_string,answer='|'.join(gold),knowledge=knowledge[0]))
            print(para_know)
            temp=dict()
            temp['prompt']=kr_prompt.format(triple=noisy_string,ques=sample["question"])
            temp["chosen"]=para_know
            temp["rejected"]=knowledge[0]
            DPO.append(temp)
            temp_record=dict()
            temp_record["question"]=sample["question"]
            temp_record["answer"]=gold
            temp_record["graph"]=sample["graph"]
            temp_record["restrict_graph"]=sample["restrict_graph"]
            temp_record["ex_graph"]=sample["ex_graph"]
            temp_record["noisy"]=noisy_string
            temp_record["know_list"]=[]
            temp_record["output_list"]=[para_know.strip(),knowledge[0].strip()]
            temp_record["ans_list"]=[]
            temp_record["prefer"]=""
            temp_record["source"]="single knowledge"
            record.append(temp_record)
            continue
        
        # compare answer entities number in different representation formats
        ans_know=[]
        for k in knowledge:
            num=0
            for a in gold:
                if a in k:
                    num+=1
            ans_know.append([num,k])
        ans_know=sorted(ans_know, key=lambda x: x[0])
        if ans_know[0][0]!=ans_know[-1][0]:
            temp=dict()
            temp['prompt']=kr_prompt.format(triple=noisy_string,ques=sample["question"])
            temp["chosen"]=ans_know[-1][1]
            temp["rejected"]=ans_know[0][1]
            DPO.append(temp)
            temp_record=dict()
            temp_record["question"]=sample["question"]
            temp_record["answer"]=gold
            temp_record["graph"]=sample["graph"]
            temp_record["restrict_graph"]=sample["restrict_graph"]
            temp_record["ex_graph"]=sample["ex_graph"]
            temp_record["noisy"]=noisy_string
            temp_record["know_list"]=[]
            temp_record["output_list"]=[ans_know[-1][1],ans_know[0][1]]
            temp_record["ans_list"]=[]
            temp_record["prefer"]=""
            temp_record["source"]="answer entity"
            record.append(temp_record)
            continue
        
        # select different enough sentence pair
        knowledge=sim_compute(knowledge)
        # original output
        output_list=[]
        # extract knowledge
        know_list=[]
        for i in knowledge:
            know=''
            if MODE=='CoT':                
                for line in i.split('\n'):
                    if not line.startswith('Reason') and len(line.strip())!=0:
                        if 'Knowledge ' in line:
                            know=know+line.replace('Knowledge ','')[3:]+'\n'
                        else:
                            know=know+line+'\n'
            if MODE=='kg-to-text':
                know=i
            know=know.strip()
            know_list.append(know)
            output_list.append(i.strip())
        
        # answer
        ans_list=[]
        for i in know_list:
            ans_list.append(getResponse(ans_prompt.format(knowledge=i.strip(),ques=sample["question"]).strip()))
        print(ans_prompt.format(knowledge=i.strip(),ques=sample["question"]).strip())
        print(ans_list)
        
        # gold number
        gold_num=[]
        for i in gold:
            if i.isdigit() and num_dict.get(i):
                gold_num.append(num_dict[i])
                
        FLAG=False
        # judge use gold_num or gold
        for i in gold_num:
            if i.lower() in ans_list[0].lower() or i.lower() in ans_list[1].lower():
                FLAG=True
                break
        if FLAG:
            gold_ans=gold_num
        else:
            gold_ans=gold            
        
        # prefer annotation base on LLM response
        prefer=getResponse(eval_prompt.format(question=sample["question"],answer='|'.join(gold_ans),response1=ans_list[0],response2=ans_list[1]))
        print(eval_prompt.format(question=sample["question"],answer='|'.join(gold_ans),response1=ans_list[0],response2=ans_list[1]))
        print(prefer)
        verdict=prefer.strip().split('\n')[-1]
        if verdict=='A' or verdict=='C':
            reject=output_list[1]
            chosen=output_list[0]
        if verdict=='B':
            reject=output_list[0]
            chosen=output_list[1]
        
        # paraphrase chosen knowledge
        para_know=getResponse(para_prompt.format(question=sample["question"],triple=gold_string,answer='|'.join(gold_ans),knowledge=chosen))
        print(para_prompt.format(question=sample["question"],triple=gold_string,answer='|'.join(gold_ans),knowledge=chosen))
        print(para_know)
        temp=dict()
        if MODE=='CoT': 
            temp['prompt']=kr_prompt.format(triple=noisy_string,ques=sample["question"])
        if MODE=='kg-to-text':
            temp['prompt']=kr_prompt.format(triple=noisy_string)
        temp["chosen"]=para_know
        temp["rejected"]=reject
        
        # record
        DPO.append(temp)
        temp_record=dict()
        temp_record["question"]=sample["question"]
        temp_record["answer"]=gold_ans
        temp_record["graph"]=sample["graph"]
        temp_record["restrict_graph"]=sample["restrict_graph"]
        temp_record["ex_graph"]=sample["ex_graph"]
        temp_record["noisy"]=noisy_string
        temp_record["know_list"]=know_list
        temp_record["output_list"]=output_list
        temp_record["ans_list"]=ans_list
        temp_record["prefer"]=prefer
        if verdict=="C":
            temp_record["source"]="knowledge equal"
        else:
            temp_record["source"]="knowledge enhance"
        record.append(temp_record)

# save
json.dump(DPO,open(DPO_data,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
json.dump(record,open(result,'w',encoding='utf-8'),ensure_ascii=False,indent=2)

# divide into train and dev
random.shuffle(DPO)
train_num=int(len(DPO)*0.9)
json.dump(DPO[:train_num],open(DPO_train,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
json.dump(DPO[train_num:],open(DPO_dev,'w',encoding='utf-8'),ensure_ascii=False,indent=2)