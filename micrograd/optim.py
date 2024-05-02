from .tensor import Value
from typing import List
import torch

class Optimizer:
    def __init__(self, parameters:List[Value]):
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
        
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, parameters:List[Value], lr:float):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr

class Adam(Optimizer):
    def __init__(self, parameters:List[Value], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0]*len(parameters)
        self.v = [0.0]*len(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        for i,p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            mhat = self.m[i] / (1 - self.beta1**self.t)
            vhat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * mhat / (vhat**0.5 + self.eps)