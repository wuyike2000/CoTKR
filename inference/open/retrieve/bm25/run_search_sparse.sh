#!/bin/bash

index_name="index_bm25"
dataset='grailqa'
output="results/${dataset}.json"

python search.py \
    --dataset ${dataset} \
    --query_data_path ../../../../subgraph/${dataset}/data/test.json \
    --index_name ${index_name} \
    --output ${output} \
    --top_k 100 \
    --k1 0.4 \
    --b 0.4 \
    --num_process 10 \
    --eval \
    --save
