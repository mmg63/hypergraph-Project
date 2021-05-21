import torch.nn as nn
import torch
import torch.tensor

x = torch.randn(10, 3, requires_grad=True)
y = torch.randn(10, 2)
print(type(x))

# ساخت شبکه تک لایه‌ای کامل
linear = nn.Linear(3, 2)
print("W: ", linear.weight)
print("bias: ",linear.bias)

# ساخت lossfunction  و optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(linear.parameters(), lr=0.01)

# forward phase
predict = linear(x)

# compute loss
loss = criterion(predict, y)
print("loss: ", loss.item())

# backward phase
loss.backward()

# print out the gradient
print("dl/dw: ", linear.weight.grad)
print("dl/db: ", linear.bias.grad)

# one step gradient descent
optimizer.step()
print("W: ", linear.weight)
print("bias: ", linear.bias)


