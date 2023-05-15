import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
import matplotlib.pyplot as plt
from six.moves import urllib
from torch.utils.data import DataLoader


opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform) #60000 dados
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform) #10000 dados

trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
testloader = DataLoader(testset, batch_size=500, shuffle=False)

class Rede(nn.Module):
    def __init__(self, input_size=784, output_size=10, layers=[120, 84]):
        super().__init__()
        self.d1 = nn.Linear(input_size, layers[0])
        self.d2 = nn.Linear(layers[0], layers[1])
        self.d3 = nn.Linear(layers[1], output_size)
    
    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x)
        x = F.log_softmax(x, dim=1)
        return x

model = Rede()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    for b, (x_train, y_train) in enumerate(trainloader):
        b+=1
        
        # Apply the model
        import pdb; pdb.set_trace()
        y_pred = model(x_train.view(100, -1))
        loss = criterion(y_pred, y_train)
        
        #calculate the number of correct prediction
        predicted = torch.max(y_pred.data, 1)[1] # the prediction that has the maximum probability
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # update parameters
        optimizer.zero_grad() # reset the gradient after each training step
        loss.backward() # backpropagation
        optimizer.step() # perform parameters update
        
        # print results
        if b%600 == 0:
            print(f'epoch: {i:2} batch: {b:4} [{100*b:6}/60000] Train loss: {loss.item():10.8f}')
        
    #Update train loss & accuracy for the epoch
    train_losses.append(loss)
    train_correct.append(trn_corr)
    
    # Run the testting batches
    with torch.no_grad(): # don't calculate gradients during testting
        for b, (x_test, y_test) in enumerate(testloader):
            
            # apply the model
            y_val = model(x_test.view(500, -1)) # flatten x_test
            
            # tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
            
            # update test loss & accuracy for the epoch
            loss = criterion(y_val, y_test)
            test_losses.append(loss)
            test_correct.append(tst_corr)


print(f'Test accuracy : {test_correct[-1].item()*100/10000:.3f}%') # test accuracy for the last epoch
            
        

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# # model = nn.Sequential(nn.Linear(784, 128),
# #                       nn.ReLU(),batch_size=64
# #                       nn.Linear(128, 64),
# #                       nn.ReLU(),
# #                       nn.Linear(64, 10)
# #                       nn.LogSoftmax(dim=1))
# #
# # criterion = nn.CrossEntropyLoss()
# #
# #
# # images, labels = next(iter(trainloader))
# #
# # images = images.view(images.shape[0], -1)
# #
# #
# # logits = model(images)
# #
# # loss = criterion(logits, labels)
# # print(loss)


# # print('Before backward pass: \n', model[0].weight.grad)
# #
# # loss.backward()
# #
# # print('After backward pass: \n', model[0].weight.grad)
# #
# #

# model = nn.Sequential(nn.Linear(784, 128),
#                       nn.ReLU(),
#                       nn.Linear(128, 64),
#                       nn.ReLU(),
#                       nn.Linear(64, 10),
#                       nn.LogSoftmax(dim=1))

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.003)
    

# epochs = 5
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#         images = images.view(images.shape[0], -1)

#         optimizer.zero_grad()

#         output = model(images)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#     else:
#         print(f"Training loss: {running_loss / len(trainloader)}")





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
# #input()




