#!/bin/bash

## General
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=2

export HOME_DIR="/home/yassir"
export PROJ_NAME="safe"
export PROJ_DIR="${HOME_DIR}/projects/${PROJ_NAME}"
export TMP_DIR="${PROJ_DIR}/tmp"
export OUTPUT_DIR="${TMP_DIR}/outputs"
export LOG_DIR="${OUTPUT_DIR}/logs"

##  Deepspeed
# export INCLUDE="localhost:0,1,2,3,4,5,6,7"
# export WANDB_API_KEY="09c6ccb3534c407d83e0268ab44c5ffcc205d298"
# export INCLUDE="compute-permanent-node-7:0,1,2,3,4,5,6,7@compute-permanent-node-586:0,1,2,3,4,5,6,7"
export INCLUDE="localhost:0,1"
#export DS_HOSTFILE=${PROJ_DIR}/${PROJ_NAME}/expts/multinode/hostfile
export DS_HOSTFILE=${PROJ_DIR}/expts/multinode/hostfile_single
export DS_CONFIG=${PROJ_DIR}/expts/multinode/deepspeed_zero3.json
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))

# Original settings
export BATCH_SIZE=100
export NUM_LABELS=1
export INC_DESC=False
export LR=1e-4
export MLM=False
export WANDB_DISABLED=False
export WANDB_SILENT=True

# Other parameters
export RUNNER="${PROJ_DIR}/expts/multinode/model_trainer.py"

source activate devenv

# --include=$INCLUDE
#deepspeed --hostfile $DS_HOSTFILE --master_port=$MASTER_PORT $RUNNER  \
deepspeed --hostfile $DS_HOSTFILE --include=$INCLUDE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $RUNNER  \
    --deepspeed  $DS_CONFIG \
    --fp16 \
    --config configmodel.json --tokenizer tokenizer.json \
    --dataset /nfs/scratch/cristian/data/debug  \
    --text_column "input" \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_labels $NUM_LABELS \
    --include_descriptors $INC_DESC \
    --do_train \
    --logging_dir $LOG_DIR \
    --streaming True \
    --gradient_accumulation_steps 2  \
    --max_steps 100 --save_total_limit 10 \
    --eval_accumulation_steps 100 --logging_steps 1 --logging_first_step True \
    --save_safetensors True --do_train True --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --warmup_steps 50000 \
    --gradient_checkpointing True \
    --save_strategy "steps"  \
    --torch_compile True  \
    --save_steps 50000 \
    --log_on_each_node True  \
    --report_to none \
    --disable_tqdm False

# --wandb_watch 'gradients'
# --wandb_project SAFE_training \
