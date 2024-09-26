import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer,AutoModel
from peft import PeftModel

# dataset: GraphQuestions, grailqa
DATA='grailqa'
# llm: llama-2-7b-chat-hf, Meta-Llama-3-8B-Instruct
LLM='llama-2-7b-chat-hf'
# mode
MODE='CoT'
# path for LLM
LLM_PATH='../../pretrain/'+LLM
# path for tokenizer
TOKENIZER_PATH='../../pretrain/'+LLM
# path for lora
PEFT_PATH='output-'+DATA+'/'+MODE+'/'+LLM+'/best_model'
# result
result='output-'+DATA+'/sft/'+MODE+'/'+LLM

tokenizer=AutoTokenizer.from_pretrained(LLM_PATH)
llm=AutoModelForCausalLM.from_pretrained(LLM_PATH,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='cuda:0')
llm=PeftModel.from_pretrained(llm, PEFT_PATH,torch_dtype=torch.float16,device_map='cuda:0')
llm=llm.merge_and_unload()
llm.save_pretrained(result)
