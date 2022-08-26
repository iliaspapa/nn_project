# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:30:27 2018

@author: Ilias
"""
#last stable version for cifer10
#beter verzion pending 
if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)
    
    classes = ('airplane', 'automobile', 'bird', 'poorkitty', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    # functions to show an image
    
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
        
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #nn.init.calculate_gain
            self.iteration = 0
            self.conv1 = nn.Conv2d(3, 128, 5, padding = 3)
            torch.nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(128, 128, 3, padding = 1)
            torch.nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
            self.conv3 = nn.Conv2d(128, 256, 3, padding = 1)
            torch.nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
            self.conv2_drop = nn.Dropout2d()
            self.conv2_drop2 = nn.Dropout2d()
            self.fc1 = nn.Linear(4096, 500)
            self.fc2 = nn.Linear(500, 100)
            self.fc3 = nn.Linear(100, 10)
        
        def forward(self, x):
            self.iteration+=1
            x = self.pool(F.relu(self.conv1(x)))
            x = self.conv2_drop2(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.conv2_drop(x)
            x = self.pool(F.relu(self.conv3(x)))
           # x = self.conv2_drop(x)
            x = x.view(-1, 4096)
            x = F.relu(self.fc1(x))
           # x = F.dropout(x, training=self.training)
            x = F.relu(self.fc2(x))
           # x = F.dropout(x, training=self.training)
            x = self.fc3(x)
            return x
        ''' 
           def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features
        '''

    
    net = Net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    maxx=0
    c=time.time()
    for epoch in range(50):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        d=time.time()-c
        print('epoch lasted %f seconds' % d)
        c=time.time()
        dataiter = iter(testloader)
        images, labels = dataiter.next()
    
        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
    
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
        correct = float(0)
        total = float(0)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        d=time.time()-c
        print('testing lasted %f seconds' % d)
        print('Accuracy of the network on the 10000 test images: %f %%' % (
                100.0 * (correct*1.0) / (1.0*total)))
        if 100.0 * (correct*1.0) / (1.0*total)>maxx:
            maxx=100.0 * (correct*1.0) / (1.0*total)
            maxpos=epoch+1
        print('best so far is: %f %% in epoch %d '% (maxx,maxpos))
        c=time.time()
    print('Finished Training')
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    correct = float(0)
    total = float(0)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100.0 * (correct*1.0) / (1.0*total)))
    if 100.0 * (correct*1.0) / (1.0*total)>maxx:
        maxx=100.0 * (correct*1.0) / (1.0*total)
        maxpos=epoch-1
    print('best is: %f %% in epoch %d '% (maxx,maxpos))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %2f %%' % (
            classes[i], 100.0 * (1.0*class_correct[i]) / (1.0*class_total[i])))