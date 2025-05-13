import torch
import torch.nn as nn

from src.models import get_results

def eval(model: nn.Module, test_dataloader: object) -> float:
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


if __name__ == '__main__':
    main()
