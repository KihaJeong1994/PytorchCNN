import torch
x= torch.rand(3,3)
y= torch.rand(3,3)

if torch.cuda.is_available():
    x = x.cuda() # send to gpu memory
    y = y.cuda()
    sum = x+y

print(sum)
print(sum.mean())
print(sum.sum())