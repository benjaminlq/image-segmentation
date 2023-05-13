import torch
import torch.nn as nn
import config
from typing import Optional
from tqdm import tqdm

import os

def calculate_score(labels, preds):
    # score = function(labels, preds)
    return

def train(
    model,
    datamanager,
    learning_rate: float = 1e-4,
    no_epochs: int = 50,
    batch_size: Optional[int] = 16,
    early_stop: bool = False,
    patience: int = 5,):

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    if batch_size:
        datamanager.batch_size = batch_size
    trainloader = datamanager.trainloader()
    valloader = datamanager.valloader()
    
    model.to(config.DEVICE)
    
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(no_epochs):
        model.train()
        tk0 = tqdm(trainloader, total=len(trainloader))
        epoch_loss = 0
        for batch_idx, (imgs, labels) in enumerate(tk0):
            imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            outs = model(imgs)
            loss = criterion(outs, labels)
            
            epoch_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            
            if (batch_idx%200) == 0:
                print(f"Epoch {epoch + 1} - Batch {batch_idx}: {loss.item}")
                
        print(f"Epoch {epoch + 1} Train Loss: {epoch_loss/len(trainloader)}")
        eval_loss, score = eval(model, valloader)
        print(f"Epoch {epoch + 1} Validation Loss: {eval_loss}")
        
        if eval_loss < best_loss:
            # save_model
            ckpt_path = os.path.join(config.ARTIFACT_PATH, "best.ckpt")
            print(f"Eval loss improved from {best_loss} to {eval_loss}. Model saved at {ckpt_path}.")
            best_loss = eval_loss
            patience_counter = 0
        
        else:
            patience_counter += 1
            if patience_counter == patience and early_stop:
                print("Early Stopped due to {} epochs without validation loss improvement.")
                break
    
def eval(model, valloader):
    model.eval()

    
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    tk0 = tqdm(valloader, total=len(valloader))
    epoch_loss = 0
    preds, labels = [], []
    for imgs, batch_labels in enumerate(tk0):
        imgs = imgs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        outs = model(imgs)
        loss = criterion(batch_labels, outs)
        
        preds.append(torch.argmax(outs, dim = 1))
        
        epoch_loss += loss.item()
        
    score = calculate_score(labels, preds)
        
    return epoch_loss / len(valloader), score