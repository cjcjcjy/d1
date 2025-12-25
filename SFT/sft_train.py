import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
from sft_trainer import *
import torch.distributed as dist
import random
import numpy as np


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="/home/jcyang/models/GSAI-ML/LLaDA-8B-Instruct", help="Name of the pretrained model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data0/devaansh",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-s1", help="Job Name")
    parser.add_argument("--train_data", type=str, default="simplescaling/s1K", help="Path to training data")
    parser.add_argument(
        "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
    )

    return parser.parse_args()


# Model loading with LoRA integration
def load_model_and_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    # Load model
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Applying LoRA model
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)  # Cast fp32 lora params to bf16

    return tokenizer, model


# Dataset loading
def load_data(args, tokenizer):
    # 本地缓存路径
    local_cache_dir = "./data"
    dataset_name = args.train_data.replace("/", "_")
    local_parquet_path = os.path.join(local_cache_dir, f"{dataset_name}.parquet")

    os.makedirs(local_cache_dir, exist_ok=True)

    if os.path.exists(local_parquet_path):
        # 从本地 Parquet 加载
        print(f"Loading dataset from local cache: {local_parquet_path}")
        data = load_dataset("parquet", data_files=local_parquet_path, split="train")
    else:
        # 首次加载，从远程下载并保存为 Parquet
        print(f"Downloading dataset from {args.train_data}...")
        data = load_dataset(args.train_data, split="train", cache_dir=local_cache_dir)
        print(f"Saving dataset to local cache: {local_parquet_path}")
        data.to_parquet(local_parquet_path)

    train_data, eval_data = preprocess_dataset(data, tokenizer, args.max_length)
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset


# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset = load_data(args, tokenizer)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=2,
        save_steps=100,
        save_total_limit=20,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    # Create optimizer and scheduler
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
    )
    print("Total training steps: ", num_train_steps)
    # Initialize Trainer with custom dLLMTrainer
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    init_seed(42)
    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args)

    # Train the model
    train_model(args, tokenizer, model)
