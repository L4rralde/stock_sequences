import math
import yaml
from glob import glob
import os

import torch
import torch.nn as nn
import numpy as np

from src.utils import GIT_ROOT

class MyFirstLstm(nn.Module):
    def __init__(self, hidden_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 9),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.lstm1(seq)
        output = self.dropout1(output)
        output, (hidden, _) = self.lstm2(output)
        seq_out = hidden[-1]
        seq_out = self.dropout2(seq_out)
        preds = self.predictor(seq_out)
        return preds


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, n_ff_layers: int=2) -> None:
        super().__init__()
    
        #Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        
        #Feedforward model
        self.ff_block = nn.ModuleList()
        self.ff_block.append(
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        )
        for _ in range(n_ff_layers - 2):
            self.ff_block.append(
            nn.Sequential(
                nn.Linear(dim_feedforward, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        )   
        self.ff_block.append(nn.Linear(dim_feedforward, d_model))

        #Normalization layers:
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    #Self-attention block
    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.ff_block:
            x = layer(x)
        return self.dropout2(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._sa_block(x) + x
        x = self.norm1(x)
        x = self._ff_block(x) + x
        x = self.norm2(x)

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


with open("params.yaml") as stream:
    CONFS = yaml.safe_load(stream)


class MyTransformerEncoderRegressor(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, max_len: int=100, ntransformers: int=1, n_ff_layers: int = 2) -> None:
        super().__init__()
        self.positional_encoding =  PositionalEncoding(d_model, max_len)
        self.transformers = nn.ModuleList()
        self.transformers.append(MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, n_ff_layers=n_ff_layers))
        for _ in range(ntransformers - 1):
            self.transformers.append(MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, n_ff_layers=n_ff_layers))
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = x[:, -1, :]
        x = self.regression_head(x)
        return x

    @classmethod
    def from_params(cls, conf: str, weights: str="best") -> "MyTransformerEncoderRegressor":
        model = cls(
            d_model = 9,
            nhead = CONFS[conf]['NHEADS'],
            dim_feedforward = CONFS[conf]['DIM_FEEDFORWARD'],
            max_len = 100,
            ntransformers = CONFS[conf]['NTRANSFORMERS'],
            n_ff_layers = CONFS[conf]['N_FF_LAYERS']
        )
        path = f"{GIT_ROOT}/models/{conf}/{weights}"
        status = model.load_state_dict(
            torch.load(
                path,
                weights_only=True,
                map_location=torch.device('cpu')
            )
        )
        #print(status)
        return model

    @staticmethod
    def get_training_hist(conf) -> dict:
        dir = f"{GIT_ROOT}/models/{conf}"
        if not os.path.exists(dir):
            raise RuntimeError("Dir does not exist")
        arrays = glob(f"{dir}/*.npy")
        hist = {
            os.path.basename(array).split('.')[0]: np.load(array)
            for array in arrays
        }
        return hist


RESULTS = dict(CONFS)
for conf in CONFS:
    model = MyTransformerEncoderRegressor.from_params(conf)
    training_history = MyTransformerEncoderRegressor.get_training_hist(conf)
    RESULTS[conf]['model'] = model
    RESULTS[conf]['training_history'] = training_history


class BaselineModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :]
