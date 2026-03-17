import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

# Assuming these are available in the finetune package
from finetune.config import TrainConfig
from finetune.lora import save_lora_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions: MultipleNegativesRankingLoss (InfoNCE) and Triplet Loss
# (Moved from train.py as they are core to the Trainer's operation)
# ---------------------------------------------------------------------------


def compute_mnrl_loss(
    query_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE / MultipleNegativesRankingLoss.

    Given a batch of (query, positive) embedding pairs:
      1. Compute cosine similarity matrix: sim[i][j] = cos(query_i, positive_j)
      2. Labels = diagonal (each query_i should match positive_i)
      3. Loss = CrossEntropy(sim / temperature, labels)

    In-batch negatives: for query_i, all positive_j (j !=