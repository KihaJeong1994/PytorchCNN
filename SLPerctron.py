import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
Y = torch.FloatTensor([[0],[1],[1],[1]]).to(device)

linear = nn.Linear(2,1,bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear,sigmoid).to(device)

# loss function & optimizer
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=1)

nb_epochs = 1000

for epoch in range(nb_epochs):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis,Y)
    cost.backward()
    optimizer.step()
    if epoch % 100 ==0:
        print("step:",epoch,"cost:",cost.item())

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis>0.5).float()
    accuracy = (predicted==Y).float().mean()
    print('hypothesis',hypothesis.detach().cpu().numpy())
    print('Predicted',predicted.detach().cpu().numpy())
    print('Y',Y.cpu().numpy())
    print('Accuracy',accuracy.item())