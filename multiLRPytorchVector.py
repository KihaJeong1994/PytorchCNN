import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train=torch.FloatTensor([
    [73,80,75],
    [93,88,93],
    [89,91,90],
    [96,98,100],
    [73,66,70],
])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# torch.Size([5,3]) # input shape
# torch.Size([5,1]) # ouput shape

W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)



# choose optimizer
optimizer = optim.SGD([W,b],lr=1e-5) # important to chooose lr
nb_epochs = 20

for epoch in range(nb_epochs+1):
    hypothesis = x_train.matmul(W) + b
    #define cost function
    cost = torch.mean((hypothesis - y_train) ** 2)
    # do not accumulate gradient
    optimizer.zero_grad()
    # gradient calculate
    cost.backward()
    #update W,b
    optimizer.step()


    print('Epoch{:4d}/{} hypothesis:{} Cost: {:.6f}'.format(epoch,
                                                                       nb_epochs, hypothesis.squeeze().detach(), cost.item()))