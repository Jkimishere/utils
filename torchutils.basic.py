import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_test_split(array, train):
    if train:
        array = array[0:int(len(array) * 0.9)] #split images(first 90%)
    else:
        array = array[int(len(array) * 0.9):] #split images(last 10%)
    
