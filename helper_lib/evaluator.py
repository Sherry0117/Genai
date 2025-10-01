import torch
from torch.nn.functional import cross_entropy


@torch.no_grad()
def evaluate_model(model, data_loader, criterion=None, device='cpu'):
    """Compute average loss and accuracy over a dataloader.

    If `criterion` is None, uses cross entropy.
    Returns (avg_loss, accuracy) where accuracy is in [0,1].
    """
    model.eval()

    if criterion is None:
        def criterion(outputs, targets):
            return cross_entropy(outputs, targets)

    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predicted = outputs.argmax(dim=1)
        total_correct += (predicted == targets).sum().item()

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)

    return avg_loss, accuracy
