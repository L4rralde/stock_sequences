import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.models import MyTransformerEncoderRegressor
from src.dataset import TRAIN_DATASET, VAL_DATASET
from src.train import train


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


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

    train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE)

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
