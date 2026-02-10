from typing import Any, Dict, List, Optional, Tuple

import collections
import math
import os
import random

import datasets
import numpy as np
import torch

from utils.tokenization import get_tokenizer_max_length


class TokenizedCollator:
    # TODO: Fix to use separate tokenizers

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        out_ex = collections.defaultdict(list)
        for ex in features:
            for col in ex:
                out_ex[col].append(ex[col])

        extra_keys = []
        for k, v in out_ex.items():
            if isinstance(v, list) and isinstance(v[0], int):
                out_ex[k] = torch.tensor(v)
            if isinstance(v, list) and isinstance(v[0], list):
                if (len(v[0]) > 0) and isinstance(v[0][0], int):
                    out_ex[k] = torch.tensor(v)
                else:
                    # skip empty lists
                    continue
            else:
                try:
                    out_ex[k] = torch.stack(v)
                except TypeError:
                    # can't stack string, etc. -- just leave em
                    extra_keys.append(k)

        return dict(out_ex)


class MultiencoderTokenizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        encoders: dict[str, Any],
        n_embs_per_batch: int,
        batch_size: int,
        max_length: int,
        seed: int = 42,
    ):
        self.tokenizers = {
            enc_name: enc.tokenizer for enc_name, enc in encoders.items()
        }
        self.tokenizer_names = list(sorted(self.tokenizers.keys()))
        self.n_embs_per_batch = n_embs_per_batch
        self.batch_size = batch_size
        self.max_length = max_length

        self.n_batches = math.ceil(len(dataset) / self.batch_size)
        self.batch_token_name_idxs = []
        for _ in range(self.n_batches):
            self.batch_token_name_idxs.append(self._get_batch_token_name_idxs())

        self._len = len(dataset)
        self.dataset = dataset
        new_indices = np.arange(len(dataset))
        # shuffle indices
        np.random.Generator(np.random.PCG64(seed)).shuffle(new_indices)
        self.new_indices = list(new_indices)

    def _get_batch_token_name_idxs(self) -> tuple[int, int]:
        return random.sample(range(len(self.tokenizers)), k=self.n_embs_per_batch)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tok_name_idxs = self.batch_token_name_idxs[idx // self.batch_size]
        tok_names = [self.tokenizer_names[i] for i in tok_name_idxs]
        tokenizer_max_lengths = [
            get_tokenizer_max_length(self.tokenizers[tok_name])
            for tok_name in tok_names
        ]
        max_length = min(*tokenizer_max_lengths, self.max_length)
        _max_num_chars = max_length * 5

        ex_text = self.dataset[int(self.new_indices[idx])]["text"]
        ex_text = ex_text[:_max_num_chars]
        output = {}
        for tok_name in tok_names:
            tt = self.tokenizers[tok_name](
                ex_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            if tok_name == "clip":
                tt["image_text_info"] = torch.tensor(1)

            output.update({f"{tok_name}_{key}": value for key, value in tt.items()})

        if "token_name_idxs" in output:
            output.pop("token_name_idxs")
        return {k: v.flatten() for k, v in output.items()}


class MultiEncoderClassificationDataset(MultiencoderTokenizedDataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        encoders: dict[str, Any],
        n_embs_per_batch: int,
        batch_size: int,
        max_length: int,
        seed: int = 42,
        return_texts: bool = False,
    ):
        super().__init__(
            dataset, encoders, n_embs_per_batch, batch_size, max_length, seed
        )
        self.labels = None
        self.label_texts = None
        self.encoders = encoders
        self.return_texts = return_texts

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = super().__getitem__(idx)
        out["label"] = self.dataset[int(self.new_indices[idx])]["label"]
        if self.return_texts:
            out["text"] = self.dataset[int(self.new_indices[idx])]["text"]
        return out


class EmbeddingAlignmentDataset(torch.utils.data.Dataset):
    """
    Pairs single-cell unlabeled embeddings with protein-labeled embeddings for alignment training.

    Assumes embeddings are already computed; this dataset handles pairing and shuffling,
    And expects a set of unlabeled embeddings and one of labeled embeddings with their labels in a 3rd file.

    """

    def __init__(
        self,
        unlabeled_embeddings: np.ndarray,  # Shape: (N, D_unlabeled)
        labeled_embeddings: np.ndarray,  # Shape: (M, D_labeled)
        protein_labels: np.ndarray,  # Shape: (M,) - protein codes
        pairing_strategy: str = "random",  # "random" or "nearest"
        seed: int = 42,
    ):
        """
        Args:
            unlabeled_embeddings: High-dim embeddings without labels
            labeled_embeddings: High-dim embeddings with protein labels
            protein_labels: Protein code identifiers for labeled embeddings
            pairing_strategy: How to match unlabeled to labeled embeddings
            seed: Random seed for reproducibility
        """
        self.rng = np.random.Generator(np.random.PCG64(seed))

        # Validate shapes
        assert (
            unlabeled_embeddings.shape[0] == labeled_embeddings.shape[0]
        ), "Both embedding sets must have same number of samples"
        assert labeled_embeddings.shape[0] == len(
            protein_labels
        ), "Number of labels must match number of labeled embeddings"

        self._len = unlabeled_embeddings.shape[0]
        self.unlabeled_embeddings = torch.from_numpy(unlabeled_embeddings).float()
        self.labeled_embeddings = torch.from_numpy(labeled_embeddings).float()
        self.protein_labels = protein_labels

        # Shuffle both embeddings consistently
        shuffle_indices = np.arange(self._len)
        self.rng.shuffle(shuffle_indices)

        self.unlabeled_embeddings = self.unlabeled_embeddings[shuffle_indices]
        self.labeled_embeddings = self.labeled_embeddings[shuffle_indices]
        self.protein_labels = self.protein_labels[shuffle_indices]

        self.pairing_strategy = pairing_strategy

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns paired embeddings with protein label for the labeled embedding.
        """
        return {
            "unlabeled_embedding": self.unlabeled_embeddings[idx],
            "labeled_embedding": self.labeled_embeddings[idx],
            "protein_label": self.protein_labels[idx],
        }


class UnpairedEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings: np.ndarray, seed: int = 42):
        self.embeddings = torch.from_numpy(embeddings).float()
        self._len = self.embeddings.shape[0]

        rng = np.random.Generator(np.random.PCG64(seed))
        indices = np.arange(self._len)
        rng.shuffle(indices)
        self.embeddings = self.embeddings[indices]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return {"unsup_emb": self.embeddings[idx]}


class LabeledEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        protein_labels: np.ndarray | None = None,
        seed: int = 42,
    ):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = (
            torch.from_numpy(protein_labels).long()
            if protein_labels is not None
            else None
        )

        self._len = self.embeddings.shape[0]

        rng = np.random.Generator(np.random.PCG64(seed))
        indices = np.arange(self._len)
        rng.shuffle(indices)

        self.embeddings = self.embeddings[indices]
        if self.labels is not None:
            self.labels = self.labels[indices]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        out = {"sup_emb": self.embeddings[idx]}
        if self.labels is not None:
            out["protein_label"] = self.labels[idx]
        return out


from torch.utils.data import Dataset


class UnsupervisedEmbeddingDataset(Dataset):
    def __init__(self, embeddings, emb_name):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.emb_name = emb_name

    def __getitem__(self, idx):
        return {self.emb_name: self.embeddings[idx]}

    def __len__(self):
        return len(self.embeddings)


class SupervisedEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, emb_name):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = labels
        self.emb_name = emb_name

    def __getitem__(self, idx):
        return {self.emb_name: self.embeddings[idx]}

    def __len__(self):
        return len(self.embeddings)


def split_dataset(dataset, val_fraction, seed):
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_size = int(n * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
    )
