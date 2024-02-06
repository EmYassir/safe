#!/bin/bash

#SBATCH --job-name=yassir-test              # name
#SBATCH --nodes=2                           # nodes
#SBATCH --mem=200G                          # memory
#SBATCH --ntasks-per-node=1                 # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16                  # number of cores per tasks
#SBATCH --gpus-per-node=A100:2              # number of gpus
#SBATCH --time 1:00:00                      # maximum execution time (HH:MM:SS)
#SBATCH --reservation=yassir                # reservation name
#SBATCH --output=/home/yassir/projects/safe/expts/multinode/test.out      # output file name
#SBATCH --error=/home/yassir/projects/safe/expts/multinode/test.out       # error file name

## General
export NODE_RANK=0
export N_NODES=2
export N_GPU_NODE=2
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=1,0
export GPUS_PER_NODE=2
#export MASTER_ADDR=172.16.5.168
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=2
export WANDB_API_KEY="09c6ccb3534c407d83e0268ab44c5ffcc205d298"


# DEBUG
#export NCCL_DEBUG=INFO
#export TORCH_CPP_LOG_LEVEL=INFO 
#export TORCH_DISTRIBUTED_DEBUG=INFO 
#export TORCH_SHOW_CPP_STACKTRACES=1
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1 


export HOME_DIR="/home/yassir"
export PROJ_NAME="safe"
export PROJ_DIR="${HOME_DIR}/projects/${PROJ_NAME}"
export TMP_DIR="${PROJ_DIR}/tmp"
export OUTPUT_DIR="${TMP_DIR}/outputs"
export LOG_DIR="${OUTPUT_DIR}/logs"

##  Deepspeed
export INCLUDE="172.16.5.168:0,1@172.16.6.139:0,1"
export DS_HOSTFILE=${PROJ_DIR}/expts/multinode/hostfile
export DS_CONFIG=${PROJ_DIR}/expts/multinode/deepspeed_zero3.json
export ACCELERATE_CFG_FILE=${PROJ_DIR}/expts/multinode/accelerate_megatron_multi_node.yaml

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
export TOKENIZER="${PROJ_DIR}/expts/multinode/tokenizer.json"



source /home/yassir/.conda_envs/devenv/bin/activate
#mamba init
#mamba activate devenv


export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --config_file $ACCELERATE_CFG_FILE \
    "
export RUNNER_ARGS=" \
    --fp16 \
    --config configmodel.json --tokenizer $TOKENIZER \
    --dataset /nfs/scratch/cristian/data/debug  \
    --text_column "input" \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_labels $NUM_LABELS \
    --include_descriptors $INC_DESC \
    --do_train \
    --logging_dir $LOG_DIR \
    --streaming True \
    --gradient_accumulation_steps 2  \
    --max_steps 1000 --save_total_limit 1 \
    --eval_accumulation_steps 100 --logging_steps 1 --logging_first_step True \
    --save_safetensors True --do_train True --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --warmup_steps 50000 \
    --gradient_checkpointing True \
    --save_strategy "steps"  \
    --torch_compile True  \
    --save_steps 50000 \
    --log_on_each_node 0  \
    --report_to none \
    --disable_tqdm False \
    --log_level info \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $RUNNER $RUNNER_ARGS" 
srun $CMD

# --wandb_watch 'gradients'
# --wandb_project SAFE_training \
