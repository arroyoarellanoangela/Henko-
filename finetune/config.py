"""Training configuration for LoRA fine-tuning."""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    # Base model
    base_model: str = "BAAI/bge-m3"
    hidden_dim: int = 1024  # output dimension of bge-m3

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16  # scaling = alpha / rank = 2.0
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")  # attention projections

    # Training
    learning_rate: float = 2e-5
    batch_size: int = 64
    gradient_accumulation_steps: int = 4  # effective batch = 256
    epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    temperature: float = 0.05  # InfoNCE temperature

    # Data
    train_data_path: Path = BASE_DIR / "data" / "finetune" / "pairs.jsonl"
    eval_split: float = 0.1
    max_seq_length: int = 128

    # Output
    output_dir: Path = BASE_DIR / "data" / "finetune" / "checkpoints"
    loss_plot_path: Path = BASE_DIR / "data" / "finetune" / "loss_curve.png"

    # Data generation
    sample_chunks: int = 3000
    questions_per_chunk: int = 2
    groq_batch_size: int = 5
    groq_delay_s: float = 3.0
