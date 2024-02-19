def training_loop(model, epochs, train_loader, val_loader, loss_fn, optimizer, model_name, scheduler=None, plot_graph : bool =True) -> None:
        print(f'starting training for model {model_name}')
        training_losses = []
        validation_losses = []
        training_start = time()
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            loss = 0.0
            epoch_start = time()
            model.train() 
            for i, data in enumerate(tqdm(train_loader, leave=False,position=1)):
                x, y = data
                out = model(x)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()
            epoch_end = time()
            if plot_graph:
                training_losses.append(loss.cpu().item())
            print(f'epoch {epoch} ended with loss {loss} ||| epoch {epoch} runtime : {epoch_end - epoch_start} seconds')
            print('starting validation')
            validation_losses.append(validation_loop(model, val_loader))
        training_end = time()
        print(f'training done in {int(training_end - training_start)} seconds, or {int(training_end - training_start) / 60} minutes')
        torch.save(model.state_dict(), f'./{model_name}.pth')
        if plot_graph:
            plt.plot(list(range(epochs)), training_losses, label='Training loss')
            plt.plot(list(range(epochs)), validation_losses, label='Validation loss')
            plt.title('Loss graph')
            plt.legend()
            plt.show()
        
        
def validation_loop(model, val_loader):
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = inputs.to(device), labels.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.cpu().item() * x.size(0)
    avg_loss = total_loss / len(val_loader.dataset)
    print(f'Average loss for model in validation is : {avg_loss}')
    return avg_loss
