#!/bin/bash

#SBATCH --job-name=yassir-deepspeed         # name
#SBATCH --nodes=2                           # nodes
#SBATCH --mem=200G                          # memory
#SBATCH --ntasks-per-node=1                 # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16                  # number of cores per tasks
#SBATCH --gpus-per-node=a100:2              # number of gpus
#SBATCH --time 1:00:00                      # maximum execution time (HH:MM:SS)
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/safe/expts/multinode/test.out      # output file name
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/safe/expts/multinode/test.out       # error file name

## General
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="ceeb005a3731a69cc1377d12184e2c28ede292bf"
export WANDB_DISABLED=false
export WANDB_SILENT=true
export WANDB_LOG_MODEL=0

## Jobs
export CUDA_VISIBLE_DEVICES=0,1
export NODE_RANK=$SLURM_NODEID
export N_NODES=$SLURM_NNODES
export WORLD_SIZE=$SLURM_NTASKS
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE#*:}
export NUM_PROCS=$((N_NODES * GPUS_PER_NODE))
#export WORLD_SIZE=$NUM_PROCS
#export WORLD_SIZE=$N_NODES
#export LOCAL_WORLD_SIZE

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))

# DEBUG
#export NCCL_DEBUG=INFO
#export TORCH_CPP_LOG_LEVEL=INFO 
#export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_P2P_DISABLE=1
#export TORCH_SHOW_CPP_STACKTRACES=1
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1

### Test if NCCL problems occur ???
#export NCCL_SOCKET_IFNAME=ens3
#export NCCL_SOCKET_IFNAME=eth4
#export NCCL_SOCKET_IFNAME=lo

# Directories
export HOME_DIR="/mnt/ps/home/CORP/yassir.elmesbahi"
export PROJ_NAME="safe"
export PROJ_DIR="${HOME_DIR}/project/${PROJ_NAME}"
export TMP_DIR="${PROJ_DIR}/tmp"
export OUTPUT_DIR="${TMP_DIR}/outputs"
export LOG_DIR="${OUTPUT_DIR}/logs"

##  Deepspeed
#export INCLUDE="172.16.5.168:0,1@172.16.6.139:0,1"
#export DS_HOSTFILE=${PROJ_DIR}/expts/multinode/hostfile
export DS_CONFIG=${PROJ_DIR}/expts/multinode/deepspeed_zero3.json
export ACCELERATE_CFG_FILE=${PROJ_DIR}/expts/multinode/accelerate_deepspeed_3.yaml

# Training settings
export BATCH_SIZE=100
export NUM_LABELS=1
export INC_DESC=False
export LR=1e-4
export MLM=False

# Other parameters
export RUNNER="${PROJ_DIR}/expts/multinode/model_trainer.py"
export TOKENIZER="${PROJ_DIR}/expts/multinode/tokenizer.json"
export DATASET_PATH="/mnt/ps/home/CORP/cristian.gabellini/project/outgoing/big_datasets/data"

source ${HOME_DIR}/.conda_envs/devenv/bin/activate

export LAUNCHER="torchrun \
--nproc_per_node $GPUS_PER_NODE \
--nnodes $N_NODES \
--node_rank $NODE_RANK \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
--rdzv_backend c10d \
--max_restarts 0 \
--tee 3 \
"

export RUNNER_ARGS=" \
    --deepspeed $DS_CONFIG \
    --fp16 \
    --config configmodel.json --tokenizer $TOKENIZER \
    --dataset $DATASET_PATH  \
    --text_column "input" \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_labels $NUM_LABELS \
    --include_descriptors $INC_DESC \
    --do_train \
    --logging_dir $LOG_DIR \
    --streaming True \
    --gradient_accumulation_steps 2  \
    --max_steps 5 --save_total_limit 1 \
    --eval_accumulation_steps 100 --logging_steps 1 --logging_first_step True \
    --save_safetensors True --do_train True --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --warmup_steps 50000 \
    --gradient_checkpointing True \
    --save_strategy "steps"  \
    --save_steps 50000 \
    --log_on_each_node 0  \
    --report_to none \
    --disable_tqdm False \
    --log_level debug \
    --torch_compile True  \
    --wandb_watch all \
    --wandb_project SAFE_training \
    "
#    --wandb_watch all \
#    --wandb_project SAFE_training \

    
# This step is necessary because accelerate launch does not handle multiline arguments properly
#echo "LAUNCHER == ${LAUNCHER}"
export CMD="${LAUNCHER} ${RUNNER} ${RUNNER_ARGS}" 
echo "===>>> Running command '${CMD}'"
srun --jobid $SLURM_JOBID --export=ALL $CMD
