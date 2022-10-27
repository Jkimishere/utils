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
def training_loop(epochs, model, trainloader,loss_fn, optimizer):
        training_start = time()
        for epoch in range(epochs):
            loss = 0.0
            print(f'epoch {epoch}')
            start = time()
            model.train() 
            for i, data in enumerate(trainloader):
                img, label = data
                if i % 100 == 0:
                    print(i)
                out = model(img)
                loss = loss_fn(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            end = time()
            print(f'epoch {epoch} ended with loss {loss} ||| epoch {epoch} runtime : {end - start} seconds')

        training_end = time()
        print(f'training done in {int(training_end - training_start)} seconds, or {int(training_end - training_start) / 60} minutes')
        torch.save(model.state_dict(), './Model.pth')



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