import torch


def dice_loss(x1, x2):
    dim = [2, 3, 4] if len(x2.shape) == 5 else [2, 3]
    inter = torch.sum(x1 * x2, dim=dim)
    union = torch.sum(x1 + x2, dim=dim)
    return 1 - (2. * inter / union).mean()
