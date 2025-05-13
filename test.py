import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import get_results, BaselineModel
from src.dataset import TEST_DATASET

def eval(model: nn.Module, test_dataloader: DataLoader) -> float:
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            y_hat = model(x)
            loss = loss_fn(
                torch.squeeze(y_hat),
                torch.squeeze(y)
            )

            batch_size = x.size(0)
            total_loss += loss.data.item() * batch_size
            num_samples += batch_size
    
    total_loss /= num_samples
    return total_loss


def main() -> None:
    results = get_results()
    test_dataloader = DataLoader(TEST_DATASET, batch_size=32)
    for conf, result in results.items():
        model = result['model']
        loss = eval(model, test_dataloader)
        print(f"{conf} loss: {loss:.3e}")
    
    default_model = BaselineModel()
    loss = eval(default_model, test_dataloader)
    print(f"Default models loss: {loss:.3e}")

if __name__ == '__main__':
    main()
