import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
import matplotlib.pyplot as plt
from six.moves import urllib
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


def get_transform():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
    return transform

def load_generic_loader(dataset, n, batch_size, random_seed):
    
    dataset_size = len(dataset) 
    indices = np.arange(dataset_size)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    new_dataset_indices, other_indices = indices[0:n], indices[n:]
    new_dataset_sampler = SubsetRandomSampler(new_dataset_indices)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=new_dataset_sampler)
    
    return trainloader

def load_trainloader(n, batch_size, random_seed=23):
    """retorna o dataset do mnist no tamanho n

    Args:
        dataset (_type_): _description_
        n (_type_): _description_
        random_seed (int, optional): _description_. Defaults to 23.
    """
    transform = get_transform()
    dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    
    return load_generic_loader(dataset, n, batch_size, random_seed)

def load_testloader(n, batch_size, random_seed=23):
    
    transform = get_transform()
    dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    
    return load_generic_loader(dataset, n , batch_size, random_seed)

def train(trainloader, testloader, epochs, n_train):
    print('\n')
    print(f'Results with n = {n_train} \n')
    
    model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)
            )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    
    for e in range(epochs):
        
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")
            
        with torch.no_grad(): # don't calculate gradients during testting
            contador = 0
            accuracy = 0
            for (x_test, y_test) in testloader:
                
                y_val = model(x_test.view(x_test.shape[0], -1)) # flatten x_test
                
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr = (predicted == y_test).sum()
                accuracy += tst_corr.item()/len(y_test)
                contador += 1
            accuracy = accuracy/contador
            print(f'Accuracy: {accuracy:.3f}')
        
    
batch_size_test = 20
batch_size_train = 50
n_train = [1000, 2500, 5000]
n_test = 500
epochs = 5

testloader = load_testloader(n=n_test, batch_size=batch_size_test)

for n in n_train:
    trainloader = load_trainloader(n=n, batch_size=batch_size_train)
    train(trainloader, testloader, epochs, n_train=n)




