import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random
from tqdm import tqdm
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer,AutoModel
import torch
from peft import PeftModel
import sys
import openai
import time
from openai import OpenAI

# generation config
generation_config = GenerationConfig(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=1024
)

# dataset: grailqa, GraphQuestions
DATA='grailqa'
# llm: llama-2-7b-chat-hf, Meta-Llama-3-8B-Instruct
LLM='Meta-Llama-3-8B-Instruct'

# set client
client=OpenAI(api_key='YOUR KEY')

test=json.load(open('../../../subgraph/'+DATA+'/gold/test.json','r',encoding='utf-8'))

kr_prompt_llm='''Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: {triple}
Question: {ques}
'''

kr_prompt_gpt='''Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: (Oxybutynin Oral, medicine.routed_drug.route_of_administration, Oral administration) (Oxybutynin Oral, medicine.routed_drug.marketed_formulations, Oxybutynin chloride 5 extended release film coated tablet) (Oxybutynin Chloride Oral, medicine.routed_drug.marketed_formulations, Oxybutynin chloride 5 extended release film coated tablet) (Oxybutynin chloride 5 extended release film coated tablet, medicine.drug_formulation.formulation_of, Oxybutynin)
Question: oxybutynin chloride 5 extended release film coated tablet is the ingredients of what routed drug?
Reason 1: I need to know which routed drug has the marketed formulation of oxybutynin chloride 5 extended release film coated tablet.
Knowledge 1: The routed drugs Oxybutynin Oral and Oxybutynin Chloride Oral have the marketed formulation of oxybutynin chloride 5 extended release film coated tablet.

Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: (Google, organization.organization.founders, Sergey Brin) (Sergey Brin, people.person.education, CVT1) (CVT1, education.education.institution, University of Maryland, College Park) (Google, organization.organization.founders, Larry Page) (Larry Page, people.person.education, CVT2) (CVT2, education.education.institution, University of Michigan) (CVT2, education.education.institution, Stanford University)
Question: where did the founder of google go to college?
Reason 1: I need to know who the founders of Google are.
Knowledge 1: The founders of Google are Sergey Brin and Larry Page.
Reason 2: I need to know where Sergey Brin and Larry Page went to college.
Knowledge 2: Sergey Brin went to the University of Maryland, College Park for college. Larry Page went to the University of Michigan and Stanford University for college.

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

num_dict = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
}
   
if LLM!='chatgpt':
    # path for LLM
    LLM_PATH='../../../instruction-tuning/output-'+DATA+'/sft/CoT/'+LLM
    # path for tokenizer
    TOKENIZER_PATH='../../../../pretrain/'+LLM
    # path for lora
    PEFT_PATH='../../../instruction-tuning/output-'+DATA+'/PA-chatgpt/CoT/'+LLM+'/best_model'
    print(PEFT_PATH)
    # load tokenizer and llm
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    llm=AutoModelForCausalLM.from_pretrained(LLM_PATH,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='cuda:0')
    # merge peft into base LLM
    if PEFT_PATH:
        llm=PeftModel.from_pretrained(llm, PEFT_PATH,torch_dtype=torch.float16,device_map='cuda:0')

# result
result='result/'+DATA+'/'+LLM+'/pa-chatgpt.json'
os.makedirs('result/'+DATA+'/'+LLM,exist_ok = True)
log_file='log/'+DATA+'/'+LLM+'/pa-chatgpt.log'
os.makedirs('log/'+DATA+'/'+LLM,exist_ok = True)

# redirect output to log
sys.stdout = open(log_file, 'w')

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
    output = tokenizer.decode(generation_output[0],skip_special_tokens=True)
    response = output.split(prompt)[-1].strip()
    return response

data=[]
for sample in tqdm(test):

    # knowledge rewriter
    if len(sample["triples"])!=0:
        if LLM!='chatgpt':
            knowledge=LLMResponse(kr_prompt_llm.format(triple=sample["triples"],ques=sample["question"]),llm,tokenizer,'cuda:0')
            print(kr_prompt_llm.format(triple=sample["triples"],ques=sample["question"]))
            print(knowledge)
        else:
            knowledge=getResponse(kr_prompt_gpt.format(triple=sample["triples"],ques=sample["question"]))
            print(kr_prompt_gpt.format(triple=sample["triples"],ques=sample["question"]))
            print(knowledge)            
    else:
        knowledge=''

    # record
    temp=dict()
    temp['question']=sample['question']
    temp['answer']=sample["answer"]
    temp['graph']=sample["triples"]
    temp['knowledge']=knowledge
    data.append(temp)

json.dump(data,open(result,'w',encoding='utf-8'),indent=2,ensure_ascii=False)