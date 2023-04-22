import torch

def dice_coeff(
    preds: torch.Tensor,
    labels: torch.Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6
):
    assert preds.size() == labels.size(), "Predictions and Labels dimensions do not match"
    assert preds.dim() == 3 or not reduce_batch_first
    
    sum_dim = (-1, -2) if preds.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    intersection = 2 * (preds * labels).sum(dim=sum_dim)
    
    
def iou(
    preds: torch.Tensor,
    labels: torch.Tensor,
)
    

def dice_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    multiclass: bool = False
)