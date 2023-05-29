import numpy as np
import torch

def get_dice_loss(preds: torch.tensor, labels: torch.tensor):
    dice_loss = None
    return dice_loss

def get_iou(preds: torch.tensor, labels: torch.tensor):
    iou = None
    return iou

def get_accuracy(preds: torch.tensor, labels: torch.tensor):
    # (bs, h, w)
    correct_pixels = (preds == labels).sum().item()
    total_pixels = preds.size(0) * preds.size(1) * preds.size(2)
    return correct_pixels / total_pixels

if __name__ == "__main__":
    sample_pred = torch.randint(0, 2, size = (16, 800, 800))
    sample_label = torch.randint(0, 2, size = (16, 800, 800))
    acc = get_accuracy(sample_pred, sample_label)
    print(acc)
    