import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train=torch.FloatTensor([
    [1,2],
    [2,3],
    [3,1],
    [4,3],
    [5,3],
    [6,2],

])
y_train = torch.FloatTensor([[0],[0],[0],[1],[1],[1]])


W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# choose optimizer
optimizer = optim.SGD([W,b],lr=1) # important to chooose lr
nb_epochs = 1000

for epoch in range(nb_epochs+1):
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    #define cost function
    cost = -(y_train*torch.log(hypothesis)+(1-y_train)*torch.log(1-hypothesis)).mean()
    # do not accumulate gradient // 0 initialization
    optimizer.zero_grad()
    # gradient calculate
    cost.backward()
    #update W,b
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch{:4d}/{} Cost: {:.6f}'.format(epoch,nb_epochs, cost.item()))

print("-------------------------")
print("-------------------------")
print("-------------------------")
hypothesis = torch.sigmoid(x_train.matmul(W)+b)
print(hypothesis)


pred_y = hypothesis >= torch.FloatTensor([0.5])

print("prediction:" ,pred_y)

print("-----parameter after training---")
print(W)
print(b)