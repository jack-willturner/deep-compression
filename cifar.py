import torch
import torchvision
import torchvision.transforms as transforms

# torchvision outputs PILImages of range [0,255] - transform them to tensors
# of range [0,1] and normalize them to [-1,1]

'''
CIFAR contains 32x32x3 coloured images

************ Compose ************
Compose acts as you would expect but in reverse order from mathematical notation.

************ ToTensor ************
Convert a PIL Image (I think this is Python Imaging Library, but no references) to a tensor
from range [0,255] to [0.0, 1.0]

************ Normalize ***********
Takes two args, a list of means and standard deviations for each dimension
I'm guessing we have RGB images so give three means/stds. The pytorch docs say we
normalize to range [-1,1] but from what I can tell we are normalizing to range [0,1].
Strange since ToTensor already does this.
'''

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


# Load up the images and create a list of classes they can fit to. Fairly straightforward
''' Is there a way to load the training/test/validation sets in one line? '''

trainset    = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset     = torchvision.datasets.CIFAR10(root='./data'. train=False, download=True, transform=transform)
testloader  = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')




# print out some images and their labels

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # unnormalize ... WAT
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0))) # WAT

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))




'''
Actual network stuff

'''

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5) ''' in_channels, out_channels, kernel_size '''
        self.pool  = nn.MaxPool2d(kernel_size=2,stride=2) ''' kernel_size, stride '''
        self.conv2 = nn.Conv2d(6,15,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) ''' in_channels, out_channels '''
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()




''' Loss / optimizer '''
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


''' Training '''
num_epochs = 2

for epoch in range(num_epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader,0)
        inputs, labels = data
        # wat
        inputs, labels = Variable(inputs), Variable(labels)

        # zero out the gradients on each training iteration
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 2000 == 1999:
            if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        print('Finished Training')







correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))




class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
