# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:42:44 2018

@author: ilias
"""
#trexei kai se cpu kai se gpu 
#PARAKALW PEIRAMATISTITE PANW
'''
to-dos
pooling-batchsize(ilias)
normalization/batchnorm(mallon galanis)
deeper-sholower
filters
activation fanctions
dropout
'''
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,#fere ta data se 1 diastasi kai kanonikopoiise sto 0-1
                                            download=True       , transform=transform)
    #print(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=1)
    
    classes = ('airplane', 'automobile', 'bird', 'poorkitty', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # functions to show an image

    def zerograd(lr):
        if epoch<lr[0]:
            optimizer[0].zero_grad()
        elif epoch<lr[1]:
            optimizer[1].zero_grad()
        elif epoch<lr[2]:
            optimizer[2].zero_grad()
        elif epoch<lr[3]:
            optimizer[3].zero_grad()
        else:
            optimizer[4].zero_grad()
    
    def optstep(lr):
        if epoch<lr[0]:#alla3e parapmetrous  
            optimizer[0].step()
        elif epoch<lr[1]:
            optimizer[1].step()
        elif epoch<lr[2]:
            optimizer[2].step()
        elif epoch<lr[3]:
            optimizer[3].step()
        else:
            optimizer[4].step()
        
    def testit(maxx, maxpos):
        c=time.time()
        net.training = False
        correct=0.0
        total=0.0
        testing_loss = 0.0
        print('testing...')
        with torch.no_grad():
            for minibatches, data in enumerate(testloader,0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)#calculate loss
                testing_loss += loss.item()
        d=time.time()-c
        print('testing lasted %f seconds' % d)
        testing_accuracy = 100.0 * (correct*1.0) / (1.0*total)
        testing_loss = testing_loss/minibatches
        print('Accuracy of the network on the 10000 test images: %f %% with average loss of %f' % (
                testing_accuracy, testing_loss))
        if testing_accuracy>maxx:
            maxx = testing_accuracy
            maxpos = epoch+1
        print('best so far is: %f %% in epoch %d '% (maxx,maxpos))
        return maxx, maxpos


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    class Net(nn.Module):#the net
        
        def __init__(self):#constraxtor....ti prepei na ginee stin arxi
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 42, 3, padding = 1)
            self.pool = nn.MaxPool2d(5, stride = 2)
            self.pool2 = nn.MaxPool2d(5, stride = 3)
            self.conv2 = nn.Conv2d(42, 42, 3, padding = 1)
            self.conv3 = nn.Conv2d(42, 42, 3, padding = 1)
            self.conv4 = nn.Conv2d(42, 42, 3, padding = 1)
            self.conv5 = nn.Conv2d(42, 42, 3, padding = 1)
            self.conv6 = nn.Conv2d(42, 84, 3, padding = 1)
            self.rrelu1_1 = nn.RReLU()
            self.rrelu1_2 = nn.RReLU()
            self.rrelu1_3 = nn.RReLU()
            self.rrelu1_4 = nn.RReLU()
            self.rrelu1_5 = nn.RReLU()
            self.rrelu1_6 = nn.RReLU()
            self.rrelu2_1 = nn.RReLU()
            self.rrelu2_2 = nn.RReLU()
            torch.nn.init.xavier_normal_(self.conv1.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv2.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv3.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv4.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv5.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.conv6.weight, gain = nn.init.calculate_gain('leaky_relu'))
            self.fc1 = nn.Linear(1344, 700)
            self.fc2 = nn.Linear(700, 200)
            self.fc3 = nn.Linear(200, 10)
            torch.nn.init.xavier_normal_(self.fc1.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.fc2.weight, gain = nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.fc3.weight, gain = nn.init.calculate_gain('leaky_relu'))
            self.iterations = 0.0
        
        def forward(self, x):#kane provlepsi
            x = self.rrelu1_1(self.conv1(x))
            x = self.rrelu1_2(self.conv2(x))
            x = self.rrelu1_3(self.conv3(x))
            x = self.pool(x)
            x = self.rrelu1_4(self.conv4(x))
            x = self.rrelu1_5(self.conv5(x))
            x = self.rrelu1_6(self.conv6(x))
            x = self.pool2(x)
            x = x.view(-1, 1344)
            x = F.dropout(x, training=self.training, p = min([self.iterations/70000.0, 0.3]))#dropout only when training
            x = self.rrelu2_1(self.fc1(x))
            x = F.dropout(x, training=self.training, p = min([self.iterations/70000.0, 0.3]))#dropout only when training
            x = self.rrelu2_2(self.fc2(x))
            #x = F.dropout(x, training=self.training, p = 0.15)
            x = self.fc3(x)
            if self.training and self.iterations<21000:#0.3*70000
                self.iterations += 1.0
                if self.iterations%1000==0:
                    print('we are in the %d th iteration' %(self.iterations))
            return x
    
    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()#loss function
    optimizer = []
    optimizer.append(optim.SGD(net.parameters(), lr=0.007, momentum=0.9, weight_decay=0.0005))#optimizers me diafwretika learning rate mporei na ginei automata
    optimizer.append(optim.SGD(net.parameters(), lr=0.0008, momentum=0.9, weight_decay=0.0005))
    optimizer.append(optim.SGD(net.parameters(), lr=0.00008, momentum=0.9, weight_decay=0.0005))
    optimizer.append(optim.SGD(net.parameters(), lr=0.0000015, momentum=0.9, weight_decay=0.0005))
    optimizer.append(optim.SGD(net.parameters(), lr=0.0000008, momentum=0.9, weight_decay=0.0005))
    
        #actual code
    maxx = 0
    maxpos = -1
    c=time.time()
    for epoch in range(40):  # loop over the dataset multiple times
        net.training = True
        total = float(0)
        correct = float(0)
        running_loss = 0.0
        average_loss = 0.0
        average_accuracy = 0.0
        supertotal = float(0)#for all photoes in epoch
        for i, data in enumerate(trainloader, 0):
            #print(data)
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            
            lr =[17, 27, 32, 39]
            zerograd(lr)
            # forward + backward + optimize
            outputs = net(inputs)#make prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)#calculate loss
            loss.backward()#backpropagate
            optstep(lr)
            #print(2)
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                training_accuracy = 100.0 * (correct*1.0) / (1.0*total)
                print('[%d, %5d]added loss: %.4f and training accuracy of %f %%' %
                      (epoch + 1, i + 1, running_loss/200, training_accuracy))
                average_loss += running_loss/200
                average_accuracy += correct
                supertotal += total
                running_loss = 0.0
                total = float(0)
                correct = float(0)
        
        
        #testing stuff for every epoch
        d=time.time()-c
        av_training_accuracy = (100.0*average_accuracy)/(1.0*supertotal)
        print('epoch lasted %f seconds and had average loss of: %f and an average training accurasy of: %f %%' % 
             (d,average_loss/15,av_training_accuracy))
        maxx, maxpos = testit(maxx, maxpos)
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
            images, labels = images.to(device), labels.to(device)
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