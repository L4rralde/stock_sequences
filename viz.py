import os

import torch
from torch.utils.data import DataLoader
from torchview import draw_graph

from src.dataset import TEST_DATASET
from src.models import RESULTS
from src.utils import GIT_ROOT



IMG_DIR = f"{GIT_ROOT}/docs/imgs"
os.makedirs(IMG_DIR, exist_ok=True)

test_dataloader = DataLoader(TEST_DATASET, batch_size=32)

for x, _ in test_dataloader:
    break
z = torch.zeros_like(x)

for conf, result in RESULTS.items():
    model = result['model']
    model_graph = draw_graph(model, z)
    fpath = f"{IMG_DIR}/{conf.lower()}"
    model_graph.visual_graph.render(fpath, format="jpeg")
