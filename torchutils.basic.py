import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_test_split(arr, train):
    if train:
        arr = arr[0:int(len(arr) * 0.9)] #split images(first 90%)
    else:
        arr = arr[int(len(arr) * 0.9):] #split images(last 10%)
    
