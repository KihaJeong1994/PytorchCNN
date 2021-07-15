import torch
from torch.autograd import Variable

x= torch.ones(2,2)
x = Variable(x,requires_grad=True) # varible for differentiation

print(x)
print(x.data)
print(x.grad)
print(x.grad_fn)

y = x+2
print(y)

z=y**2
print(z)

out = z.sum()
print(out)
out.backward() # backward() => gradient calculate

print("--x.data--")
print(x.data)
print("--x.grad--")
print(x.grad)
print("--x.grad_fn--")
print(x.grad_fn)

# print(y.data)
# print(y.grad)
# print(y.grad_fn)
#
# print(z.data)
# print(z.grad)
# print(z.grad_fn)
#
# print(out.data)
# print(out.grad)
# print(out.grad_fn)