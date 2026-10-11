"""Single-GPU assistant-only SFT, adapted from the historical Trainer entry.

This is a runnable migration, not a reconstruction of the unknown historical
launch command. Learning rate and initialization model must be supplied.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from agentic_text2sql.sft.masking import tokenize_with_mask

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project-name",
        default="verl_text2sql_agent",
    )
    parser.add_argument(
        "--run-name",
        default="agentic_text2sql_sft",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
    )

    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    args = parser.parse_args(argv)
    positive = (
        "max_length", "epochs", "learning_rate", "micro_batch_size",
        "gradient_accumulation_steps", "logging_steps", "eval_steps", "save_steps",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(float(value)) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args

class AgentSFTDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        tokenizer,
        max_length: int,
    ) -> None:
        from datasets import load_dataset

        raw_dataset = load_dataset(
            "parquet",
            data_files=str(parquet_path),
            split="train",
        )

        self.items = []

        for row in raw_dataset:
            messages = json.loads(row["messages_json"])
            tools = json.loads(row["tools_json"])

            item = tokenize_with_mask(
                tokenizer,
                messages,
                tools,
                max_length,
            )
            item["trajectory_id"] = row["trajectory_id"]
            self.items.append(item)

        if not self.items:
            raise RuntimeError(
                f"Empty dataset: {parquet_path}"
            )

        supervised = sum(
            sum(label != -100 for label in item["labels"])
            for item in self.items
        )
        total = sum(
            len(item["input_ids"])
            for item in self.items
        )

        print(
            f"{parquet_path.name}: "
            f"rows={len(self.items)}, "
            f"tokens={total}, "
            f"supervised_tokens={supervised}, "
            f"supervised_ratio={supervised / total:.4%}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]

        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "attention_mask": item["attention_mask"],
        }

class AgentSFTCollator:
    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        multiple = self.pad_to_multiple_of
        max_length = (
            (max_length + multiple - 1)
            // multiple
            * multiple
        )

        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_length),
            -100,
            dtype=torch.long,
        )

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])

            input_ids[row, :length] = torch.tensor(
                feature["input_ids"],
                dtype=torch.long,
            )
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(
                feature["labels"],
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def validate_run_paths(args):
    """Validate a fresh, single-process run before loading a tokenizer or model."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This entry supports a single process only")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU through CUDA_VISIBLE_DEVICES")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    for path in (args.train_file, args.validation_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.train_file.resolve() == args.validation_file.resolve():
        raise ValueError("Training and validation files must differ")
    if args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            raise RuntimeError("Output directory must be absent or empty")
    output = args.output_dir.resolve()
    for path in (args.model_path, args.train_file, args.validation_file):
        resolved = path.resolve()
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise RuntimeError("Output path must not overlap model or input paths")

def main() -> None:
    args = parse_args()
    validate_run_paths(args)
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed,
    )
    set_seed(args.seed)
    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.project_name

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = AgentSFTDataset(
        args.train_file,
        tokenizer,
        args.max_length,
    )
    validation_dataset = AgentSFTDataset(
        args.validation_file,
        tokenizer,
        args.max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("总参数量：", total_parameters)
    print("训练参数量：", trainable_parameters)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=(
            args.micro_batch_size
        ),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[] if args.report_to == "none" else ["wandb"],
        run_name=args.run_name,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": AgentSFTCollator(
            tokenizer.pad_token_id
        ),
    }

    trainer_signature = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()

    final_dir = args.output_dir / "final_hf"
    model.config.use_cache = True

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_dataset)
    metrics["validation_samples"] = len(
        validation_dataset
    )

    (
        args.output_dir / "training_summary.json"
    ).write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print("最终HF模型：", final_dir)
    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )

    if args.report_to == "wandb":
        import wandb
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()
