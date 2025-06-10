#!/bin/bash

# Update your filepath 
cd /home/reubenlee3/workspace/case-studies/deep_propensity_model

python train.py \
--data_dir /home/reubenlee3/workspace/case-studies/deep_propensity_model \
--user_history_length 5 \
--batch_size 32 \
--user_embedding_size 64 \
--product_embedding_size 64 \
--hidden_size 128 \
--num_epochs 1