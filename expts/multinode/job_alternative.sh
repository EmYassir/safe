#!/usr/bin/env bash

## Files for logs: here we redirect stoout and sterr to the same file

#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gres=gpu:A100:8            # number of gpus
#SBATCH --time 20:00:00                              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/test-nodes.out           # output file name
#SBATCH --error=logs/test-nodes.err            # output file name
#SBATCH --reservation=safe

set -e

source activate safetraining 

#export WANDB_PROJECT="SAFE_SPACE"
# Execute your workload
# cd /home/nikhil_valencediscovery_com/projects/openMLIP
# srun python src/mlip/train.py experiment=drugs-chem-hyp-v2/mace-small/1m-n32.yaml

# A dummy and useless `sleep` to give you time to see your job with `squeue`.
# select data where hydra config wills search
# export DATA_DIR="/home/cristian_valencediscovery_com/dev/openMLIP/expts/"
#wandb offline #online
# Other parameters
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9920
export RUNNER=/home/cristian/dev/safe/expts/multinode/model_trainer.py
export DS_CONFIG=/home/cristian/dev/safe/expts/multinode/deepspeed_stage3.json
export BATCH_SIZE=100

echo "SLURM_JOB_UID=$SLURM_JOB_UID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    $RUNNER  \
    --deepspeed $DS_CONFIG \
    --fp16 \
    --config configmodel.json --tokenizer tokenizer.json \
    --dataset /nfs/scratch/cristian/data/debug  \
    --text_column "input" \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_labels 1 --include_descriptors False \
    --do_train True  --logging_dir logs/ \
    --wandb_project SAFE_training \
    --streaming True \
    --gradient_accumulation_steps 2  --wandb_watch 'gradients' \
    --max_steps 750000 --save_total_limit 10 \
    --eval_accumulation_steps 100 --logging_steps 500 --logging_first_step True \
    --save_safetensors True --do_train True --output_dir output/safe/model_train_deepspeed/ \
    --learning_rate 1e-4 --warmup_steps 50000 --gradient_checkpointing True \
    --save_strategy "steps"  \
    --torch_compile True --report_to wandb --save_steps 50000 \
    --log_on_each_node False '