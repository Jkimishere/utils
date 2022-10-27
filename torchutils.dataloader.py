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



#dataloader for files like (category).(number).(extension) {example: dog.1.jpg}
class category_in_filename_data_loader(data.Dataset):
    def __init__(self, root, transforms, train):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imgs = os.listdir(root)
        import random
        random.shuffle(self.imgs) #shuffled the files because os.listdir is alphabetically ordered
        if train:
            self.imgs = self.imgs[0:int(len(self.imgs) * 0.9)] #split images(first 90%)
        else:
            self.imgs = self.imgs[int(len(self.imgs) * 0.9):] #split images(last 10%)
        self.train = train


    #length function
    def __len__(self):
        return len(self.imgs)

    #get item function
    def __getitem__(self,idx):
        img_loc = os.path.join(self.root, self.imgs[idx])
        file_split = self.imgs[idx].split('.')
        label = file_split[0]
        if label.lower() == 'dog':
            label = 0
        else:
            label = 1
        #opening image and applying transforms
        image = Image.open(img_loc).convert("RGB") 
        tensor_image = self.transforms(image)
        tensor_label = torch.tensor(label)

        return tensor_image.to('cuda'), tensor_label.to('cuda')







