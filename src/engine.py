import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
from torch.optim 

def train(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: Callable,
    trainloader: DataLoader,
    valloader: DataLoader, 
    args,
    ):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in args.no_epochs:
        model.train()
        epoch_loss = 0
        tk_train = tqdm(trainloader, total=len(trainloader))
        for i, (images, masks) in tk_train:
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            
            optimizer.zero_grad()
            loss = loss_fn(preds, masks)
            
            epoch_loss += loss.item()
            loss.backward()
            
            if (i%100) == 0:
                print("Something")
        
            optimizer.step()
            
            
            
def eval(
    
)