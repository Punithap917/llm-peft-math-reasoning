"""
Fine-tune and evaluate an LLM with a parameter-efficient adapter (LoRA or AdapterP)
on mathematical reasoning.

This script wraps `finetune.py` and `evaluate.py` from the course codebase
(https://github.com/duyngtr16061999/AI6130_Assignment2, based on LLM-Adapters),
so that every experiment in the report can be reproduced with a single command.

Examples
--------
# Exp 5 & 6: OpenLLaMA-3B + LoRA on math_7k, evaluated on AQuA and AddSub
python run_experiment.py --model openllama-3b --adapter lora

# Optional experiment: 3 epochs
python run_experiment.py --model openllama-3b --adapter lora --epochs 3

# Optional experiment: larger fine-tuning set
python run_experiment.py --model openllama-3b --adapter lora --data math_10k

# Print the commands without running anything
python run_experiment.py --model tinyllama --adapter adapterp --dry_run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/duyngtr16061999/AI6130_Assignment2.git"

MODELS = {
    "tinyllama": "TinyLlama/TinyLlama_v1.1",
    "openllama-3b": "openlm-research/open_llama_3b_v2",
}

# adapter key -> (finetune.py flags, evaluate.py --adapter value)
ADAPTERS = {
    "lora": (["--adapter_name", "lora"], "LoRA"),
    "adapterp": (["--adapter_name", "bottleneck", "--use_adapterp"], "AdapterP"),
}

PACKAGES = ["fire", "datasets", "transformers==4.38.0", "accelerate==0.28.0"]


def run_cmd(cmd, cwd=None, dry_run=False):
    print("\n" + "=" * 100)
    print("RUNNING:", " ".join(str(c) for c in cmd))
    print("CWD:    ", cwd or os.getcwd())
    print("=" * 100, flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo(repo_dir: Path, dry_run: bool):
    if repo_dir.exists():
        print(f"Codebase found: {repo_dir}")
    else:
        run_cmd(["git", "clone", REPO_URL, str(repo_dir)], dry_run=dry_run)


def install_packages(dry_run: bool):
    for pkg in PACKAGES:
        run_cmd([sys.executable, "-m", "pip", "install", pkg], dry_run=dry_run)
    # The codebase ships its own modified `peft`, so the pip version must be removed.
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "peft"], dry_run=dry_run)


def check_gpu():
    try:
        import torch

        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except ImportError:
        print("PyTorch not installed yet.")


def main():
    p = argparse.ArgumentParser(description="PEFT fine-tuning + evaluation for math reasoning")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--adapter", choices=ADAPTERS, required=True)
    p.add_argument("--data", default="math_7k", help="fine-tuning set in ft-training_set/ (math_7k, math_10k, ...)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval", nargs="+", default=["AQuA", "AddSub"],
                   help="benchmarks: AQuA AddSub MultiArith SingleEq gsm8k SVAMP")
    p.add_argument("--lr", default="3e-4")
    p.add_argument("--batch_size", default="4")
    p.add_argument("--micro_batch_size", default="4")
    p.add_argument("--cutoff_len", default="256")
    p.add_argument("--val_set_size", default="120")
    p.add_argument("--repo_dir", default="AI6130_Assignment2", help="where the course codebase lives")
    p.add_argument("--output_dir", default=None, help="adapter weights folder (auto-named if empty)")
    p.add_argument("--install", action="store_true", help="install pinned packages first")
    p.add_argument("--skip_train", action="store_true", help="only evaluate existing weights")
    p.add_argument("--dry_run", action="store_true", help="print commands without running them")
    args = p.parse_args()

    os.environ["WANDB_MODE"] = "disabled"

    base_model = MODELS[args.model]
    train_flags, eval_adapter = ADAPTERS[args.adapter]
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = args.output_dir or f"./trained_models/{args.model}-{args.adapter}-{args.data}-ep{args.epochs}"

    print(f"Model: {base_model} | Adapter: {eval_adapter} | Data: {args.data} | Epochs: {args.epochs}")
    print(f"Weights: {output_dir}")
    check_gpu()
    ensure_repo(repo_dir, args.dry_run)
    if args.install:
        install_packages(args.dry_run)

    if not args.skip_train:
        run_cmd(
            [sys.executable, "finetune.py",
             "--base_model", base_model,
             "--data_path", f"./ft-training_set/{args.data}.json",
             "--output_dir", output_dir,
             *train_flags,
             "--batch_size", args.batch_size,
             "--micro_batch_size", args.micro_batch_size,
             "--num_epochs", str(args.epochs),
             "--learning_rate", args.lr,
             "--cutoff_len", args.cutoff_len,
             "--val_set_size", args.val_set_size],
            cwd=repo_dir, dry_run=args.dry_run,
        )

    for dataset in args.eval:
        run_cmd(
            [sys.executable, "evaluate.py",
             "--adapter", eval_adapter,
             "--dataset", dataset,
             "--base_model", base_model,
             "--lora_weights", output_dir],
            cwd=repo_dir, dry_run=args.dry_run,
        )

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
