import torch

x = torch.ones(2,2, requires_grad=True)
y= x*x*2
z=y.mean()
z.backward()
print(x,y,z,x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x,y)
print(x.grad,y.grad)

