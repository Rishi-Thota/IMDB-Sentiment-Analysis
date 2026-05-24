
# handles all the data stuff - loading, cleaning, vocab, dataloaders
# basically everything you need before you can actually train anything

import os
import re
import json

import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def normalize_review(raw: str) -> str:
    # get rid of html tags first, there are a lot of them in this dataset
    raw = re.sub(r"<.*?>", " ", raw)
    # drop any urls
    raw = re.sub(r"http\S+", " ", raw)
    # only keep actual letters, everything else becomes a space
    raw = re.sub(r"[^a-zA-Z\s]", " ", raw)
    # squish multiple spaces down to one
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.lower()


# ---------------------------------------------------------------------------
# Vocab / Token Map
# ---------------------------------------------------------------------------

class TokenMap:
    # slots 0 and 1 are always reserved, don't touch them
    BLANK   = "<PAD>"
    MISSING = "<UNK>"

    def __init__(self, capacity: int = 25_000):
        self.capacity  = capacity
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    def fit(self, corpus: list[str]) -> "TokenMap":
        # count every word across all training docs
        freq = Counter()
        for doc in corpus:
            freq.update(doc.split())

        # pad goes to 0, unk goes to 1, then fill in by frequency
        self.token2id = {self.BLANK: 0, self.MISSING: 1}
        for tok, _ in freq.most_common(self.capacity - 2):
            self.token2id[tok] = len(self.token2id)

        self.id2token = {i: t for t, i in self.token2id.items()}
        return self

    def transform(self, doc: str) -> list[int]:
        # anything we haven't seen before gets mapped to UNK
        fallback = self.token2id[self.MISSING]
        return [self.token2id.get(tok, fallback) for tok in doc.split()]

    def dump(self, filepath: str) -> None:
        # save to json so we don't have to rebuild every time
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.token2id, fh)

    def restore(self, filepath: str) -> "TokenMap":
        with open(filepath, "r", encoding="utf-8") as fh:
            self.token2id = json.load(fh)
        self.id2token = {i: t for t, i in self.token2id.items()}
        return self

    def __len__(self) -> int:
        return len(self.token2id)


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def fix_lengths(encoded: list[list[int]], max_len: int = 256) -> np.ndarray:
    # preallocate with zeros (zeros == PAD index, so shorter seqs are fine)
    out = np.zeros((len(encoded), max_len), dtype=np.int64)
    for i, seq in enumerate(encoded):
        n = min(len(seq), max_len)
        out[i, :n] = seq[:n]
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    # nothing fancy, just wraps the numpy arrays as tensors

    def __init__(self, token_matrix: np.ndarray, sentiments: np.ndarray):
        self.token_matrix = torch.tensor(token_matrix, dtype=torch.long)
        self.sentiments   = torch.tensor(sentiments,   dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sentiments)

    def __getitem__(self, pos: int):
        return self.token_matrix[pos], self.sentiments[pos]


# ---------------------------------------------------------------------------
# Main function - call this to get your loaders
# ---------------------------------------------------------------------------

def build_loaders(
    root: str        = "data",
    vocab_limit: int = 25_000,
    seq_len: int     = 256,
    batch: int       = 64,
    seed: int        = 42,
):
    """
    Loads the IMDB csv, cleans everything up, builds the vocab from train only
    (important - don't let val/test leak into the vocab), and hands back three
    DataLoaders plus the token map in case you need it later.

    split is 80/10/10 train/val/test, stratified so class balance is preserved.
    """
    csv_path = os.path.join(root, "IMDB Dataset.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"couldn't find the dataset at '{csv_path}'\n"
            "grab 'IMDB Dataset.csv' from Kaggle and drop it in "
            f"'{root}/' - check readme.txt if you're not sure how"
        )

    print(f"[dataset] loading {csv_path} ...")
    frame = pd.read_csv(csv_path)

    print("[dataset] cleaning reviews ...")
    frame["clean"] = frame["review"].apply(normalize_review)

    # 1 for positive, 0 for negative
    frame["label"] = (frame["sentiment"] == "positive").astype(int)

    all_texts  = frame["clean"].tolist()
    all_labels = frame["label"].values

    # first cut: hold out 20% for val+test
    tr_texts, tmp_texts, tr_labels, tmp_labels = train_test_split(
        all_texts, all_labels,
        test_size=0.2, random_state=seed, stratify=all_labels
    )
    # split the holdout 50/50 into val and test
    val_texts, te_texts, val_labels, te_labels = train_test_split(
        tmp_texts, tmp_labels,
        test_size=0.5, random_state=seed, stratify=tmp_labels
    )

    print(f"[dataset] train: {len(tr_texts)}  val: {len(val_texts)}  test: {len(te_texts)}")

    # build vocab on training data only
    tok_map = TokenMap(capacity=vocab_limit)
    tok_map.fit(tr_texts)
    print(f"[dataset] vocab size: {len(tok_map)}")

    # encode then pad/truncate to fixed length
    tr_mat  = fix_lengths([tok_map.transform(t) for t in tr_texts],  seq_len)
    val_mat = fix_lengths([tok_map.transform(t) for t in val_texts], seq_len)
    te_mat  = fix_lengths([tok_map.transform(t) for t in te_texts],  seq_len)

    train_loader = DataLoader(ReviewDataset(tr_mat,  tr_labels),  batch_size=batch, shuffle=True)
    val_loader   = DataLoader(ReviewDataset(val_mat, val_labels), batch_size=batch)
    test_loader  = DataLoader(ReviewDataset(te_mat,  te_labels),  batch_size=batch)

    return train_loader, val_loader, test_loader, tok_map
