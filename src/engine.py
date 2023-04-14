import torch
import configs
from typing import Optional
from tqdm import tqdm

def segment_loss(preds, labels):
    return
    
def train(
    model,
    datamanager,
    learning_rate: float = 1e-4,
    no_epochs: int = 50,
    batch_size: Optional[int] = 16):

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)
    
    if batch_size:
        datamanager.batch_size = batch_size
    trainloader = datamanager.trainloader()
    valloader = datamanager.valloader()
    
    model.to(configs.DEVICE)
    
    for epoch in range(no_epochs):
        model.train()
        tk0 = tqdm(trainloader, total=len(trainloader))
        epoch_loss = 0
        for batch_idx, (imgs, labels) in enumerate(tk0):
            imgs = imgs.to(configs.DEVICE)
            labels = labels.to(configs.DEVICE)
            
            optimizer.zero_grad()
            outs = model(imgs)
            loss = segment_loss(outs, labels)
            
            epoch_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            
            if (batch_idx%200) == 0:
                print(f"Epoch {epoch + 1} - Batch {batch_idx}: {loss.item}")
                
        print(f"Epoch {epoch + 1} Loss: {epoch_loss/len(trainloader)}")
        eval_loss, score = eval(model, valloader)
        
    
def eval(model, valloader):
    model.eval()
    tk0 = tqdm(valloader, total=len(valloader))
    
    for imgs, labels in enumerate(tk0):
        imgs = imgs.to(configs.DEVICE)
        labels = labels.to(configs.DEVICE)
        model
        
    return 