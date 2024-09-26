llm=Meta-Llama-3-8B-Instruct
data=GraphQuestions
MODE=CoT
dataset=${data}/PA-chatgpt/${MODE}/${llm}
load_in_kbits=16
train_file=$dataset/train.json
validation_file=$dataset/dev.json
gpu_id='0'
train_batch_size=1
eval_batch_size=1
accumulation_steps=128
epoch=10
node=1
max_prompt_length=2048
max_target_length=2048
max_seq_length=4096

lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_dropout=0.05
pretrained_model=output-${data}/sft/${MODE}/${llm}
chinese_tokenizer_path=../../pretrain/${llm}
per_device_train_batch_size=${train_batch_size}
per_device_eval_batch_size=${eval_batch_size}
gradient_accumulation_steps=${accumulation_steps}
output_dir=output-$dataset
modules_to_save="embed_tokens,lm_head"
deepspeed_config_file=ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --master_port 29920 --nnodes 1 --nproc_per_node ${node} run_dpo.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs ${epoch} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --evaluation_strategy epoch \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_prompt_length ${max_prompt_length} \
    --max_target_length ${max_target_length} \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --save_safetensors False \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --load_in_kbits ${load_in_kbits} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end True \
    --report_to none 
