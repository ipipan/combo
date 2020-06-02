import torch
import torch.nn.functional as F


def masked_cross_entropy(pred: torch.Tensor, true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    pred = pred + (mask.float().unsqueeze(-1) + 1e-45).log()
    return F.cross_entropy(pred, true, reduction="none") * mask
