import torch
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

def training_loop(model, epochs, train_loader, val_loader, loss_fn, optimizer, model_name, device="cpu", scheduler=None, plot_graph: bool = True) -> None:
    
    print(f'Starting training for model: {model_name} on {device}')
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': []}
    training_start = time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time()
        
        # Training Phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validation_loop(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        epoch_end = time()
        print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_end - epoch_start:.2f}s")

    # Cleanup and Save
    total_time = time() - training_start
    print(f'Training complete in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    torch.save(model.state_dict(), f'./{model_name}.pth')

    if plot_graph:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Loss History: {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def validation_loop(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * x.size(0)
            
    return total_loss / len(val_loader.dataset)
