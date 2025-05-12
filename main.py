import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.models import MyTransformerEncoderLayer, PositionalEncoding
from src.dataset import StockDataset
from src.train import train


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    

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


dataset = StockDataset.sequential_ds()
dataset = StockDataset.item_ds()

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)


with open("params.yaml") as stream:
    d = yaml.safe_load(stream)

for k, params in d.items():
    print(k, params)
    BATCH_SIZE = params["BATCH_SIZE"]
    NHEADS = params["NHEADS"]
    DIM_FEEDFORWARD = params["DIM_FEEDFORWARD"]
    NTRANSFORMERS = params["NTRANSFORMERS"]
    N_FF_LAYERS = params["N_FF_LAYERS"]
    EPOCHS = params["EPOCHS"]
    SUBDIR = k

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MyTransformerEncoderRegressor(
        d_model=9,
        nhead=3,
        dim_feedforward=DIM_FEEDFORWARD,
        max_len=100,
        ntransformers=NTRANSFORMERS,
        n_ff_layers=N_FF_LAYERS
    )

    model = model.to(device)

    hist = train(
        model=model,
        epochs=EPOCHS,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        subdir=SUBDIR
    )
