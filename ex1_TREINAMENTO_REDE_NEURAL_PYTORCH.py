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
    other_sample = SubsetRandomSampler(other_indices)
    
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=new_dataset_sampler)
    # other = DataLoader(dataset, batch_size=batch_size, sampler=other_sample)
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

def train(trainloader, testloader, epochs, batch_size_train, batch_size_test):
    
    model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)
            )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    test_losses = []
    test_correct = []
    
    for e in range(epochs):
        tst_corr = 0
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
            for b, (x_test, y_test) in enumerate(testloader):
                
                y_val = model(x_test.view(batch_size_test, -1)) # flatten x_test
                
                # tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test).sum()
                
                # update test loss & accuracy for the epoch
                loss = criterion(y_val, y_test)
                test_losses.append(loss)
                test_correct.append(tst_corr.item()/len(y_test)/epochs)
        
    
    
    
    
    
 
        
batch_size_test = 20
batch_size_train = 50
n_train = 1000
n_test = 500
epochs = 5

trainloader = load_trainloader(n=n_train, batch_size=batch_size_train)
testloader = load_testloader(n=n_test, batch_size=batch_size_test)
train(trainloader, testloader, epochs, batch_size_train, batch_size_test)

import pdb; pdb.set_trace()

    




# images, labels = next(iter(trainloader))

# img = images[0].view(1, 784)
# label = labels[0]
# with torch.no_grad():
#     logps = model(img)


# ps = torch.exp(logps)
# plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
# plt.show()
# print(ps)
# print(label)
#input()




