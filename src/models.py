import math

import torch
import torch.nn as nn

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
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int) -> None:
        super().__init__()
    
        #Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        
        #Feedforward model
        self.ff_block = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, d_model),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #nn.Linear(dim_feedforward, d_model),
        )

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
        x = self.ff_block(x)
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
        #x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        x = x + self.pe[:, : x.size(1)]
        return x


class MyTransformerEncoderRegressor(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, max_len: int=100) -> None:
        super().__init__()
        self.positional_encoding =  PositionalEncoding(d_model, max_len)
        self.transformer_encoder = MyTransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        #x = x[:, -1, :]
        #x = self.regression_head(x)
        return x