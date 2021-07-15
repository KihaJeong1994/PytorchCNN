import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

m = nn.Conv2d(16,33,3,stride=2)
m = nn.Conv2d(16,33,(3,5),stride=(2,1), padding=(4,2))
m = nn.Conv2d(16,33,(3,5),stride=(2,1), padding=(4,2),dilation=(3,1))

input = torch.randn(20,16,50,100)
output = m(input)
# print(output)

input = torch.ones(1,1,5,5)
input = Variable(input, requires_grad = True)
filter = nn.Conv2d(1,1,3, bias=None)
print("-----filter.weight-----")
print(filter.weight)
filter.weight = nn.Parameter(torch.ones(1,1,3,3)+1)
print("-----filter.weight-----")
print(filter.weight)
out = filter(input)
print("------out-----")
print(out)