import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train =[
    [1,2,1,1],
    [2,1,3,2],
    [3,1,3,4],
    [4,1,5,5],
    [1,7,5,5],
    [1,2,5,6],
    [1,6,6,6],
    [1,7,7,7],
]
y_train = [2,2,2,1,1,1,0,0]

x_train=torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)


# one hot encoding ***
y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1,y_train.unsqueeze(1),1)

W = torch.zeros((4,3),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# model define
# model = nn.Linear(4,3)
model = nn.Sequential(
    nn.Linear(4,3),
    #nn.Softmax(1)
)
print(list(model.parameters()))

# choose optimizer
optimizer = optim.SGD(model.parameters(),lr=1e-5) # important to chooose lr
nb_epochs = 20

for epoch in range(nb_epochs+1):
    prediction = model(x_train)

    #define cost function
    cost = F.cross_entropy(prediction,y_train)
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

# new_var = torch.FloatTensor([73,80,75])
#
# pred_y = model(new_var)
#
# print("prediction:" ,pred_y)
#
# print("-----parameter after training---")
# print(list(model.parameters()))