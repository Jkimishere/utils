#pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, datasets
#other imports
import os
from PIL import Image
from time import time



#training loop
def training_loop(model, epochs, trainloader, loss_fn, optimizer, scheduler, plot_graph : bool) -> None:
        print('training start')
        print(len(trainloader))
        tl = []
        training_start = time()
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            loss = 0.0
            start = time()
            model.train() 
            for i, data in enumerate(tqdm(trainloader, leave=False,position=1)):
                img, label = data
                out = model(img)
                loss = loss_fn(out, label)
                if plot_graph:
                    tl.append(loss.cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()
            end = time()
            print(f'epoch {epoch} ended with loss {loss} ||| epoch {epoch} runtime : {end - start} seconds')
            print('starting validation')
            validation_loop()
        training_end = time()
        print(f'training done in {int(training_end - training_start)} seconds, or {int(training_end - training_start) / 60} minutes')
        torch.save(model.state_dict(), './Model.pth')
        if plot_graph:
            plt.figure(figsize=(10,5))
            plt.title("Training Loss")
            plt.plot(tl,label="train")
            plt.plot(vl, label='validation loss')
            plt.plot(va, label = 'validation f1 score')
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.show()
            plt.legend()


#testing loop
def testing_loop(model,testloader):
        model.load_state_dict(torch.load('./Model.pth'))
        correct = 0
        total = 0
        model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += (predicted == labels).sum().item()
                
                
                
vl = []
va = []
vf1 = []       

def validation_loop(model, valloader,loss_fn, is_loss, is_accuracy, is_f1):
    torch.inference_mode()
    for i, data in enumerate(valloader):
        img, label = data
        out = model(img)
        pred = out.cpu().detach().numpy()
        target = label.cpu().detach().numpy()
        if is_loss:
            #use loss function
            loss = loss_fn(out, label)
            vl.append(loss.cpu().item())
        if is_accuracy:
            #use accuracy
            va.append(accuracy_score(target,pred))
        if is_f1:
            vf1.append(f1_score(target,pred))
