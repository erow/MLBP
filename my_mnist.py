import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import *
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

import os

if __name__ == "__main__":
    cuda = False
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    net = Net().to(device)
    if os.path.exists("net"):
        checkpoint = torch.load("net")
        net.load_state_dict(checkpoint)
    else:
        data = training_set_image()
        target = training_set_label()
        data, target = data.to(device), target.to(device)
        # create your optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

        batch_size = 29999
        for epoch in range(1, 5):
            net.train()
            match = 0
            for batch_idx in range(0,int(len(data)/batch_size)):
                optimizer.zero_grad()   # zero the gradient buffers
                output = net(data[batch_idx * batch_size:batch_idx * batch_size+batch_size])
                loss = F.nll_loss(output, target[batch_idx * batch_size:batch_idx * batch_size+batch_size])
                loss.backward()
                optimizer.step()        # Does the update
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # epoch, batch_idx * batch_size, len(data),
                # 100. * batch_idx*batch_size / len(data), loss.item()))
                # 计算训练集的准确度
                _, predicted = torch.max(output.data, 1)
                match += (predicted==target[batch_idx * batch_size:batch_idx * batch_size+batch_size]).sum().item()
            print("Train Epoch: {} accuracy:{} ".format(epoch,match/len(data)))
        torch.save(net.state_dict(),"net")


    test = test_set_image()
    test_label = test_set_label()
    total = test.size(0)
    test, test_label = test.to(device), test_label.to(device)
    match = 0
    batch_size = 400
    for batch_idx in range(0,int(len(test)/batch_size)):
        result = net(test[batch_idx * batch_size:batch_idx * batch_size+batch_size])
        _, predicted = torch.max(result.data, 1)
        match += (predicted==test_label[batch_idx * batch_size:batch_idx * batch_size+batch_size]).sum().item()
    print("accuracy: ",match/total)


