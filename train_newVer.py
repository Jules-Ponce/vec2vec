import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

# from eval import eval_model
from utils.collate import (
    MultiencoderTokenizedDataset,
    TokenizedCollator,
    EmbeddingAlignmentDataset,
)
from utils.eval_utils import EarlyStopper, eval_loop_
from utils.gan import LeastSquaresGAN, RelativisticGAN, VanillaGAN
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch
from utils.train_utils import rec_loss_fn, trans_loss_fn, vsp_loss_fn, get_grad_norm
from utils.wandb_logger import Logger

from typing import Literal

DataType = Literal["archaea", "flowers", "insects", "mammals", "vertebrae"]
ALLOWED_DATATYPES = ("archaea", "flowers", "insects", "mammals", "vertebrae")


def main():
    """
    Based on the original vec2vec paper, we set up the configuration off of the config file.
    The model is then trained on our custom dataset class defined in utils/collate.py.
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f"configs/{argv[1]}.toml")
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(
        **{**{k: v for d in cfg.values() for k, v in d.items()}, **unknown_cfg}
    )

    if cfg.datatype not in ALLOWED_DATATYPES:
        raise ValueError(
            f"Invalid datatype '{cfg.datatype}'. "
            f"Expected one of: {', '.join(ALLOWED_DATATYPES)}"
        )

    if (
        hasattr(cfg, "mixed_precision")
        and cfg.mixed_precision != "no"
        and cfg.mixed_precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        cfg.mixed_precision = "fp16"
        cfg.gradient_accumulation_steps = 1
        print(
            "Note: bf16 is not available on this hardware! Reverting to fp16 and setting accumulation steps to 1."
        )

    # Set seeds for reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    accelerator = accelerate.Accelerator(
        mixed_precision=(
            cfg.mixed_precision if hasattr(cfg, "mixed_precision") else None
        ),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    # Load precomputed embeddings and labels
    unlabeled_embs = np.load(
        "data/1000_sampled_sc_embeddings.npy"
    )  # Shape: (N, D_unlabeled=512)
    labeled_embs = np.load(
        "data/mammals/1000_sampled_protein_embeddings.npy"
    )  # Shape: (M, D_labeled=1024)
    protein_labels = np.load(
        "data/mammals/1000_sampled_protein_labels.npy", allow_pickle=True
    )  # Shape: (M,)

    # Initialize the dataset and DataLoader
    dataset = EmbeddingAlignmentDataset(
        unlabeled_embeddings=unlabeled_embs,
        labeled_embeddings=labeled_embs,
        protein_labels=protein_labels,
        pairing_strategy="random",  # Default pairing strategy
        seed=cfg.seed,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=min(os.cpu_count(), 8),
        pin_memory=True,
    )

    # Initialize the model (vec2vec)
    model = LeastSquaresGAN(
        input_dim_unlabeled=unlabeled_embs.shape[1],
        input_dim_labeled=labeled_embs.shape[1],
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(cfg.save_dir, "tensorboard_logs"))

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            unlabeled_embedding = batch["unlabeled_embedding"].to(
                device
            )  # Shape: (batch_size, D_unlabeled)
            labeled_embedding = batch["labeled_embedding"].to(
                device
            )  # Shape: (batch_size, D_labeled)
            protein_label = batch[
                "protein_label"
            ]  # Not used in training, but available if needed

            # Forward pass
            loss = model(unlabeled_embedding, labeled_embedding)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Scheduler step
        scheduler.step()

        # Log epoch loss
        print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch, "loss": epoch_loss})
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        # Save model checkpoint
        if epoch % cfg.save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.save_dir, f"vec2vec_epoch_{epoch}.pt"),
            )

    writer.close()


if __name__ == "__main__":
    main()
