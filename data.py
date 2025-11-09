# src/data.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

DEFAULT_SAMPLE = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die—to sleep,\n"
)

def ensure_data_file(path="data/text.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            # write a repeated sample so data is not tiny
            f.write(DEFAULT_SAMPLE * 200)
        print(f"[data] Sample text created at {path}")

class CharDataset(Dataset):
    def __init__(self, data_path="data/text.txt", seq_len=128):
        ensure_data_file(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        # build vocab
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        # encode all
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len
        # number of samples (start positions)
        self.num_samples = max(1, len(self.data) - seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # return input_ids, target_ids (next token LM)
        start = idx
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start+1:end+1]
        return x, y

def get_dataloaders(data_path="data/text.txt", seq_len=128, batch_size=32, split=0.9):
    ds = CharDataset(data_path, seq_len)
    n = len(ds)
    n_train = int(n * split)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    from torch.utils.data import Subset
    train_ds = Subset(ds, train_idx if len(train_idx)>0 else [0])
    val_ds = Subset(ds, val_idx if len(val_idx)>0 else [0])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    vocab = {"stoi": ds.stoi, "itos": ds.itos, "vocab_size": ds.vocab_size}
    return train_loader, val_loader, vocab
