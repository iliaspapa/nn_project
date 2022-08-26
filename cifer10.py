# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:42:44 2018

@author: ilias
"""
#TO AVERAGE ACCURACY DEN DOULEUEI
#SIGOURA EPIDEXETE ARKETES VELTIWSEIS
#PARAKALW PEIRAMATISTITE PANW
#activanion
#pooling
#normalization/batchnorm
#multiprossesing
if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    #import torch.multiprocessing as mult
    import time
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #print(mult.get_all_sharing_strategies)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,#fere ta data se 1 diastasi kai kanonikopoiise sto 0-1
                                            download=True       , transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=1)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=1)
    
    classes = ('airplane', 'automobile', 'bird', 'poorkitty', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    # functions to show an image
    
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    class Net(nn.Module):#the net
        def __init__(self):#constraxtor....ti prepei na ginee stin arxi
            super(Net, self).__init__()
            #nn.init.calculate_gain
            #self.iteration = 0
            self.conv1 = nn.Conv2d(3, 32, 5, padding = 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 32, 3, padding = 1)
            self.conv3 = nn.Conv2d(32, 32, 3, padding = 1)
            self.conv4 = nn.Conv2d(32, 32, 3, padding = 1)
            self.conv5 = nn.Conv2d(32, 32, 3, padding = 1)
            self.conv6 = nn.Conv2d(32, 64, 3, padding = 1)
            self.prelu1_1 = nn.PReLU(init = 0.5)
            self.prelu1_2 = nn.PReLU(init = 0.5)
            self.prelu1_3 = nn.PReLU(init = 0.5)
            self.prelu1_4 = nn.PReLU(init = 0.5)
            self.prelu1_5 = nn.PReLU(init = 0.5)
            self.prelu1_6 = nn.PReLU(init = 0.5)
            self.prelu2_1 = nn.PReLU(init = 0.5)
            self.prelu2_2 = nn.PReLU(init = 0.5)
            torch.nn.init.xavier_normal_(self.conv1.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv2.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv3.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv4.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv5.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv6.weight, gain = nn.init.calculate_gain('leaky_relu'))
            self.fc1 = nn.Linear(1024, 500)
            self.fc2 = nn.Linear(500, 100)
            self.fc3 = nn.Linear(100, 10)
            torch.nn.init.xavier_normal_(self.fc1.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.fc2.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.fc3.weight, gain = nn.init.calculate_gain('leaky_relu'))
        
        def forward(self, x):#kane provlepsi
            #self.iteration+=1
            x = self.pool(self.prelu1_2(self.conv2(self.prelu1_1(self.conv1(x)))))
            #x = self.conv2_drop(x)
            x = self.pool(self.prelu1_4(self.conv4(self.prelu1_3(self.conv3(x)))))
            #x = self.conv2_drop(x)
            x = self.pool(self.prelu1_6(self.conv6(self.prelu1_5(self.conv5(x)))))
            x = F.dropout2d(x, training=self.training)#dropout only when training
            x = x.view(-1, 1024)
            #x = F.dropout(x, training=self.training)#dropout only when training
            x = self.prelu2_1(self.fc1(x))
            x = F.dropout(x, training=self.training)#dropout only when training
            x = self.prelu2_2(self.fc2(x))
            x = self.fc3(x)
            return x
    
    net = Net()
    
    criterion = nn.CrossEntropyLoss()#loss function
    optimizer = optim.SGD(net.parameters(), lr=0.0045, momentum=0.9, weight_decay=0.0005)#optimizers me diafwretika learning rate mporei na ginei automata
    optimizer2 = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    optimizer3 = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9, weight_decay=0.0005)
    optimizer4 = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0005)
    optimizer5 = optim.SGD(net.parameters(), lr=0.000005,momentum=0.9, weight_decay=0.0005)
    
    #actual code
    maxx = 0
    total = float(0)
    supertotal = float(0)#for all photoes in epoch
    correct = float(0)
    c=time.time()
    for epoch in range(40):  # loop over the dataset multiple times
        net.training = True
        running_loss = 0.0
        average_loss = 0.0
        average_accuracy = 0.0
        supertotal = float(0)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
    
            # zero the parameter gradients
            if epoch<11:
                optimizer.zero_grad()
            else:
                if epoch<21:
                    optimizer2.zero_grad()
                else:
                    if epoch<30:
                        optimizer3.zero_grad()
                    else:
                        if epoch<39:
                            optimizer4.zero_grad()
                        else:
                            optimizer5.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)#make prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)#calculate loss
            loss.backward()#backpropagate
            if epoch<11:#alla3e parapmetrous  
                optimizer.step()
            else:
                if epoch<21:
                    optimizer2.step()
                else:
                    if epoch<30:
                        optimizer3.step()
                    else:
                        if epoch<39:
                            optimizer4.step()
                        else:
                            optimizer5.step()
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d]added loss: %.4f and training accuracy of %f %%' %
                      (epoch + 1, i + 1, running_loss/200, 100.0 * (correct*1.0) / (1.0*total)))
                average_loss += running_loss/200
                average_accuracy += correct
                supertotal += total
                running_loss = 0.0
                total = float(0)
                correct = float(0)
        #testing stuff for every epoch
        d=time.time()-c
        print('epoch lasted %f seconds and had average loss of: %f and an average training accurasy of: %f %%' % 
             (d,average_loss/15,(100.0*average_accuracy)/(1.0*supertotal)))
        supertotal=float(0)
        c=time.time()
        net.training = False
        print('testing...')
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
    #bulshit we dont really need
    print('Finished Training')
    print('testing...')
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(16)))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(16):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print('best is: %f %% in epoch %d '% (maxx,maxpos))
    for i in range(10):
        print('Accuracy of %5s : %2f %%' % (
            classes[i], 100.0 * (1.0*class_correct[i]) / (1.0*class_total[i])))