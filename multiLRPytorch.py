import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x1_train = torch.FloatTensor([[73],[93],[89],[96],[73]])
x2_train = torch.FloatTensor([[80],[88],[91],[98],[66]])
x3_train = torch.FloatTensor([[75],[93],[90],[100],[70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# choose optimizer
optimizer = optim.SGD([w1,w2,w3,b],lr=1e-5) # important to chooose lr
nb_epochs = 10000

for epoch in range(nb_epochs+1):
    hypothesis = w1 * x1_train +w2 * x2_train +w3 * x3_train + b
    #define cost function
    cost = torch.mean((hypothesis - y_train) ** 2)
    # do not accumulate gradient
    optimizer.zero_grad()
    # gradient calculate
    cost.backward()
    #update W,b
    optimizer.step()

    if epoch %1000 ==0:
        print('Epoch{:4d}/{} w1:{:.3f} w2:{:.3f} w3:{:.3f} , b: {:.3f} Cost: {:.6f}'.format(epoch,
                                                                       nb_epochs, w1.item(), w2.item(),w3.item(),b.item(), cost.item()))