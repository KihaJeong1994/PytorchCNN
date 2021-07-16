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

# model define
model = nn.Linear(3,1)

print(list(model.parameters()))

# choose optimizer
optimizer = optim.SGD(model.parameters(),lr=1e-5) # important to chooose lr
nb_epochs = 5000

for epoch in range(nb_epochs+1):
    prediction = model(x_train)

    #define cost function
    cost = F.mse_loss(prediction,y_train)
    # do not accumulate gradient // 0 initialization
    optimizer.zero_grad()
    # gradient calculate
    cost.backward()
    #update W,b
    optimizer.step()

    if epoch % 1000 == 0:
        print('Epoch{:4d}/{} Cost: {:.6f}'.format(epoch,nb_epochs, cost.item()))

print("-------------------------")
print("-------------------------")
print("-------------------------")

new_var = torch.FloatTensor([73,80,75])

pred_y = model(new_var)

print("prediction:" ,pred_y)

print("-----parameter after training---")
print(list(model.parameters()))