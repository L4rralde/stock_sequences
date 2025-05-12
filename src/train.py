import os
import logging
from time import perf_counter

from tqdm import tqdm
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np

from src.utils import GIT_ROOT


MODELS_PATH = f"{GIT_ROOT}/models"
os.makedirs(MODELS_PATH, exist_ok=True)


def train(model: nn.Module, epochs: int, device: str, train_dataloader: object, val_dataloader, subdir: str) -> dict:
    model_dir = f"{MODELS_PATH}/{subdir}"
    os.makedirs(model_dir, exist_ok=True)


    fileh = logging.FileHandler(f"{model_dir}/training.log", 'a')
    sys_out = logging.StreamHandler()
    logger = logging.getLogger(__name__)  # root logger
    logger.setLevel(logging.INFO)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(sys_out)
    logger.addHandler(fileh)

    train_size = len(train_dataloader)
    val_size = len(val_dataloader)

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    start = perf_counter()
    for epoch in tqdm(range(1, epochs+1)):
        train_loss = 0.0
        model.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(torch.squeeze(y_hat), torch.squeeze(y))
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * x.size(0)
        train_loss /= train_size
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = loss_fn(torch.squeeze(y_hat), torch.squeeze(y))
                val_loss += loss.data.item() * x.size(0)
        val_loss /= val_size
        val_losses.append(val_loss)

        logger.info(f"Epoch: {epoch}. Training loss:{train_loss: .3e}; Validation loss: {val_loss :.3e}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"{model_dir}/{type(model).__name__}_{val_loss}"
            logger.info(f"Saving model at: {model_path}")
            torch.save(model.state_dict(), model_path)
    end = perf_counter()

    execution_time = end - start
    logger.info(f"Execution time: {execution_time: .4f}s")
    hist = {
        "training_losses": train_losses,
        "validation_losses": val_losses,
    }

    logger.info("Saving training history")
    for k, array in hist.items():
        np.save(f"{model_dir}/{k}.npy", array)
    return hist
