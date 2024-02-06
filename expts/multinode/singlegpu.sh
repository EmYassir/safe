#!/usr/bin/env bash

#SBATCH --job-name=safe_opt2

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=logs/test_single.out
#SBATCH --error=logs/test_single.err

##SBATCH --mem=5GB
##SBATCH --nodes=1    
#SBATCH --cpus-per-task=32
##SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1


set -e

# The below env variables can eventually help setting up your workload.
echo "SLURM_JOB_UID=$SLURM_JOB_UID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


source activate safetraining

#export WANDB_PROJECT="SAFE_SPACE"
# Execute your workload
# cd /home/nikhil_valencediscovery_com/projects/openMLIP
# srun python src/mlip/train.py experiment=drugs-chem-hyp-v2/mace-small/1m-n32.yaml

# A dummy and useless `sleep` to give you time to see your job with `squeue`.
# select data where hydra config wills search
# export DATA_DIR="/home/cristian_valencediscovery_com/dev/openMLIP/expts/"
#wandb offline #online

export RUNNER=/home/cristian/dev/safe/expts/multinode/model_trainer.py
export DS_CONFIG=/home/cristian/dev/safe/expts/multinode/deepspeed_stage3.json
export BATCH_SIZE=100

srun $RUNNER  \
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
    --max_steps 10000 --save_total_limit 10 \
    --eval_accumulation_steps 100 --logging_steps 500 --logging_first_step True \
    --save_safetensors True --do_train True --output_dir output/safe/model_train_deepspeed/ \
    --learning_rate 1e-4 --warmup_steps 50000 --gradient_checkpointing True \
    --save_strategy "steps"  \
    --torch_compile True --report_to wandb --save_steps 50000 