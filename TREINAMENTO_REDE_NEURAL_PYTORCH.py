import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
import matplotlib.pyplot as plt



from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# model = nn.Sequential(nn.Linear(784, 128),
#                       nn.ReLU(),batch_size=64
#                       nn.Linear(128, 64),
#                       nn.ReLU(),
#                       nn.Linear(64, 10)
#                       nn.LogSoftmax(dim=1))
#
# criterion = nn.CrossEntropyLoss()
#
#
# images, labels = next(iter(trainloader))
#
# images = images.view(images.shape[0], -1)
#
#
# logits = model(images)
#
# loss = criterion(logits, labels)
# print(loss)


# print('Before backward pass: \n', model[0].weight.grad)
#
# loss.backward()
#
# print('After backward pass: \n', model[0].weight.grad)
#
#

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
    

epochs = 5
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





images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
label = labels[0]
with torch.no_grad():
    logps = model(img)


ps = torch.exp(logps)
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
plt.show()
print(ps)
print(label)
#input()




