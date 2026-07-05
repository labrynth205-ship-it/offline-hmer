"""
Configuration for CoMER HMER pipeline.
Based on: https://github.com/Green-Wood/CoMER
Adapted for single-GPU (RTX 5060 Ti 16GB).
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Data and preprocessing configuration."""
    comer_data_dir: str = "D:/Workplace/CoMER/data"

    raw_dir: str = "dataset/raw"
    processed_dir: str = "dataset/processed"

    vocab_path: str = "dataset/processed/vocab_comer.json"

    img_channels: int = 1
    h_lo: int = 16
    h_hi: int = 128
    w_lo: int = 16
    w_hi: int = 512

    max_seq_len: int = 200

    batch_size: int = 64
    max_batch_pixels: int = 4_000_000
    num_workers: int = 4
    pin_memory: bool = True

    augment: bool = True
    scale_aug: bool = True
    scale_lo: float = 0.7
    scale_hi: float = 1.4


@dataclass
class ModelConfig:
    """CoMER model configuration (encoder + decoder)."""
    d_model: int = 256

    growth_rate: int = 24
    num_layers: int = 16

    nhead: int = 8
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.3

    dc: int = 32
    cross_coverage: bool = True
    self_coverage: bool = True

    beam_size: int = 10
    max_len: int = 200
    alpha: float = 1.0
    early_stopping: bool = False
    temperature: float = 1.0


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 300

    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4

    patience: int = 10
    lr_factor: float = 0.5

    val_every_n_epoch: int = 1

    grad_clip: float = 5.0

    use_amp: bool = True

    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    log_interval: int = 50

    output_dir: str = "outputs"


@dataclass
class Config:
    """Full configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    device: str = "cuda"

    seed: int = 7

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.output_dir, exist_ok=True)
