import os


import transformers
from safe.trainer.cli import ModelArguments
from safe.trainer.cli import DataArguments
from safe.trainer.cli import TrainingArguments
from safe.trainer.cli import train
import wandb
import torch
#torch._dynamo.config.optimize_ddp = False
print(f"#################### HEEEEEEEEERRRRRRREEEE")
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()

print(f"#################### START TRAINING")

train(model_args, data_args, training_args)
