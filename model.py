import os
import sys
sys.path.append('./')
import numpy as np
import torch
import pandas as pd
import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
from sklearn.model_selection import train_test_split
from preprocess import data_loader

## config ##
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
input_dim = len(cols)
hidden_dim = 12
output_dim = 2
EPOCH = 100
device = torch.device('cpu')
## config ##


# input data
# x = np.random.randn(100,9)
# y = np.random.randint(0,2,(100))
# print(x.shape)
# print(y.shape)

x, y = data_loader(use_cols)

print('--get data--')
print(x.shape)
print('ex)',x[0])
print('--')
print(y.shape)
print('ex)',y[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)



# create a dataloader for training
train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
# create a dataloader for validation
test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)


class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        features = self.fc3(features)
        return features

# model instance
model = Net(input_dim,hidden_dim,output_dim).to(device)
# Loss function
criterion = nn.CrossEntropyLoss().to(device)
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print()
print('Training..')
for epoch in range(1,EPOCH+1):
    print('====================================')
    print('Epoch',epoch)
    epoch_loss = 0
    for data in train_loader:
        # initialize optimizer
        optimizer.zero_grad()

        # get the inputs
        input_tensor , target_tensor = data
        # print('--input--')
        # print(input_tensor)
        # print(target_tensor)
        input_tesnor = Variable(input_tensor.to(device))
        target_tesnor = Variable(target_tensor.to(device))

        # comput scores
        score = model(input_tensor)
        # print('--output--')
        # print(score)

        # comput loss
        loss = criterion(score, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print('Epoch Loss:', epoch_loss/len(train_loader))
    
    total = 0
    correct = 0
    for data in test_loader:
        # get the inputs
        input_tensor , target_tensor = data
        input_tesnor = Variable(input_tensor).to(device)
        target_tesnor = Variable(target_tensor).to(device)
        with torch.no_grad():
            score = model(input_tensor)
            predict = F.log_softmax(score,dim=1)
            predict = torch.argmax(predict,dim=1)
            # ここらへん微妙
            bool_tensor = (predict == target_tensor)
            bool_tensor = bool_tensor.cpu().numpy()
            total += len(bool_tensor)
            corecct += np.count_nonzero(bool_tensor)
    print('Epoch Accu:',100 * correct/total)







