# import torch
# import torch.nn as nn

# loss = nn.CrossEntropyLoss()
# batch_size = 3
# n_classes = 2
# input = torch.randn(batch_size, n_classes, requires_grad=True)
# target = torch.empty(batch_size, dtype=torch.long).random_(n_classes)
# output = loss(input, target)
# output.backward()
# print(input)
# print(target)
# print(output)
# print(input.grad)

"""
tensor([[-2.6831,  1.1662],
        [ 0.9542,  0.3282],
        [-0.3262, -0.7125]], requires_grad=True)
tensor([1, 0, 0])
tensor(0.3227, grad_fn=<NllLossBackward0>)
tensor([[ 0.0069, -0.0069],
        [-0.1161,  0.1161],
        [-0.1349,  0.1349]])
"""

import micrograd.nn as nn
from micrograd import Value

loss = nn.CrossEntropyLoss()
xs = [[-2.6831,  1.1662],
        [ 0.9542,  0.3282],
        [-0.3262, -0.7125]]
input = []
for x in xs:
    input.append([Value(xi) for xi in x])
target = [1, 0, 0]
output = loss(input, target)
output.backward()
print(input)
print(output)

"""
[[Value(data=-2.6831, grad=0.006950210281274854), Value(data=1.1662, grad=-0.006950210281274849)], 
[Value(data=0.9542, grad=-0.11613935933805172), Value(data=0.3282, grad=0.11613935933805172)], 
[Value(data=-0.3262, grad=-0.1348694389274447), Value(data=-0.7125, grad=0.13486943892744468)]]
Value(data=0.3226530068216712, grad=1.0)
"""