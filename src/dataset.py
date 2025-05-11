import torch
from torch.utils.data import Dataset
import pandas as pd

from src.utils import GIT_ROOT


df = pd.read_csv(f"{GIT_ROOT}/data/data.csv")
df_filled = df.ffill()

class StockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, T: int=100, last_only: bool=False):
        self.df = df.drop(columns='Unnamed: 0')
        self.normed_df = (self.df - self.df.mean())/self.df.std()
        self.data = self.normed_df.to_numpy(dtype="float32")
        self.data = torch.from_numpy(self.data)
        self.T = T
        self._last_only = last_only

    def __len__(self):
        return len(self.df) - self.T

    def __getitem__(self, idx):
        in_seq = self.data[idx: idx+self.T]
        if self._last_only:
            out_seq = self.data[idx+self.T: idx+self.T+1]
        else:
            out_seq = self.data[idx+1: idx+self.T+1]
        return in_seq, out_seq

    @classmethod
    def sequential_ds(cls) -> "StockDataset":
        return cls(df_filled)

    @classmethod
    def item_ds(cls) -> "StockDataset":
        return cls(df_filled, last_only=True)
