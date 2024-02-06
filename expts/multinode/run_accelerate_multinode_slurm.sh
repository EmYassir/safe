#!/bin/bash

#SBATCH --job-name=yassir-multinode-safe
#SBATCH --nodes=2
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=large
#SBATCH --gpus-per-node=a100:2
#SBATCH --gpus-per-task=a100:2
#SBATCH --time 1:00:00
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/safe/expts/multinode/test.out
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/safe/expts/multinode/test.out

## General
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="09c6ccb3534c407d83e0268ab44c5ffcc205d298"
export WANDB_DISABLED=false
export WANDB_SILENT=true

## Jobs
export CUDA_VISIBLE_DEVICES=0,1
export NODE_RANK=$SLURM_NODEID
export N_NODES=$SLURM_NNODES
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE#*:}
export NUM_PROCS=$((N_NODES * GPUS_PER_NODE))
#export WORLD_SIZE=$NUM_PROCS
#export WORLD_SIZE=$N_NODES

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))


# DEBUG
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO 
export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_P2P_DISABLE=1
export TORCH_SHOW_CPP_STACKTRACES=1
#export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE=1

### Test if NCCL problems occur ???
#export NCCL_SOCKET_IFNAME=lo
#export NCCL_SOCKET_IFNAME=ens3
#export NCCL_SOCKET_IFNAME=eth4

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
export ACCELERATE_CFG_FILE=${PROJ_DIR}/expts/multinode/accelerate_fsdp_multi_node.yaml

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

export LAUNCHER="accelerate launch \
    --num_processes $NUM_PROCS \
    --num_machines $SLURM_NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --rdzv_backend c10d \
    --debug true \
    --distributed_type FSDP \
    --downcast_bf16 'no' \
    --dynamo_backend TENSORRT \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_forward_prefetch false \
    --fsdp_offload_params false \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_sync_module_states true \
    --fsdp_transformer_layer_cls_to_wrap GPT2Block \
    --fsdp_use_orig_params true \
    --mixed_precision 'fp16' \
    --same_network false \
    --use_cpu false \
    "

export RUNNER_ARGS=" \
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
    --save_strategy 'steps'  \
    --torch_compile True  \
    --save_steps 50000 \
    --log_on_each_node 0  \
    --report_to none \
    --disable_tqdm False \
    --log_level debug \
    "
# --wandb_watch 'gradients'
# --wandb_project SAFE_training \
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
echo "#########################################################"
echo ">>>>>> HOME_DIR = ${HOME_DIR}"
echo ">>>>>> PROJ_NAME = ${PROJ_NAME}"
echo ">>>>>> PROJ_DIR = ${PROJ_DIR}"
echo ">>>>>> TMP_DIR = ${TMP_DIR}"
echo ">>>>>> OUTPUT_DIR = ${OUTPUT_DIR}"
echo ">>>>>> LOG_DIR = ${LOG_DIR}"
echo ">>>>>> DS_CONFIG = ${DS_CONFIG}"
echo ">>>>>> ACCELERATE_CFG_FILE = ${ACCELERATE_CFG_FILE}"
echo ">>>>>> RUNNER = ${RUNNER}"
echo ">>>>>> TOKENIZER = ${TOKENIZER}"
echo ">>>>>> DATASET_PATH = ${DATASET_PATH}"

##echo "LAUNCHER == ${LAUNCHER}"
export CMD="${LAUNCHER} ${RUNNER} ${RUNNER_ARGS}" 
echo "===>>> Running command '${CMD}'"
srun --jobid $SLURM_JOBID --export=ALL $CMD
