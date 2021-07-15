import torch
import numpy as np
t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.]) # CPU
#t = torch.cuda.FloatTensor([0.,1.,2.,3.,4.,5.,6.]) # CPU
print(t)
print(t.dim())
print(t.shape)
print(t.size())

tr1 = torch.rand(3)
tr2 = torch.randn(3)

print(tr1)
print(tr2)

# np to torch
n1 = np.array([1,2,3,4])
t1 = torch.Tensor(n1)
print(t1)

# torch to np
t2 = torch.rand(3,3)
n2 = t2.numpy()
print(n2)

# tensor concat
t3 = torch.randn(1,1,3,3)
t4 = torch.randn(1,1,3,3)
t5 = torch.cat((t3,t4),0)
print(t5)

# tensor view (reshape)a