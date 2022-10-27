# pytorch utilities


torchutils.dataloader.py

    class category_in_filename_data_loader(data.Dataset)

    dataloader for files like (category).(number).(extension) {example: dog.1.jpg}

    use example:
        trainset = category_in_filename_data_loader('../train', transforms=transform, train= True)
        trainloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=64, 
                                         shuffle=True, num_workers=4)




torchutils.loops.py

    training_loop(epochs, model, trainloader,loss_fn, optimizer)
    basic training loop that prints out loss and time for each epoch

    <=====================================================================>

    def testing_loop(model,testloader):
    basic testing loop



torchutils.basic.py

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set device to cuda or cpu, device-agnostic code

    <=====================================================================>

    def train_test_split(array, train)
    train test split, 9 : 1 ratio

