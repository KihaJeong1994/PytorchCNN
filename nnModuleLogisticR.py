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

# model define
model = nn.Sequential( # link several models
    nn.Linear(2,1),
    nn.Sigmoid()
)

print(list(model.parameters()))

# choose optimizer
optimizer = optim.SGD(model.parameters(),lr=1)
nb_epochs = 1000

for epoch in range(nb_epochs+1):
    hypothesis = model(x_train)

    #define cost function
    cost = F.binary_cross_entropy(hypothesis,y_train)
    # do not accumulate gradient // 0 initialization
    optimizer.zero_grad()
    # gradient calculate
    cost.backward()
    #update W,b
    optimizer.step()

    if epoch % 100 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])

        correct_prediction = prediction.float() == y_train

        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print('Epoch{:4d}/{} Cost: {:.6f} Accuracy:{:2.2f}%'.format(epoch,nb_epochs, cost.item(),accuracy*100))

print("-------------------------")
print("-------------------------")
print("-------------------------")

