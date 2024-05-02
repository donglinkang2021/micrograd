from .tensor import Value
import random
from typing import List

class Module:
    def parameters(self):
        return []
    
    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)
    
class Neuron(Module):

    def __init__(self, in_features:int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
        self.b = Value(0.0)

    def forward(self, x) -> Value:
        return sum([wi * xi for wi,xi in zip(self.w, x)], self.b)

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron({len(self.w)})"
    
class Linear(Module):

    def __init__(self, in_features, out_features):
        self.neurons = [Neuron(in_features) for _ in range(out_features)]

    def forward(self, x:List[Value]) -> List[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class ReLU(Module):

    def forward(self, x:List[Value]) -> List[Value]:
        return [xi.relu() for xi in x]

    def __repr__(self) -> str:
        return "ReLU"
    
class Tanh(Module):

    def forward(self, x:List[Value]) -> List[Value]:
        return [xi.tanh() for xi in x]

    def __repr__(self) -> str:
        return "Tanh"
    
class MSELoss(Module):

    def forward(self, input:List[Value], target:List) -> Value:
        assert len(input) == len(target)
        return sum((yout - ytar)**2 for yout,ytar in zip(input, target)) / len(input)

    def __repr__(self) -> str:
        return "MSELoss"
    
class LogSoftmax(Module):

    def forward(self, x:List[Value]) -> List[Value]:
        logsumexp = sum(xi.exp() for xi in x).log()
        return [xi - logsumexp for xi in x]

    def __repr__(self) -> str:
        return "LogSoftmax"
    
class CrossEntropyLoss(Module):

    def __init__(self):
        self.logsoftmax = LogSoftmax()

    def forward(self, input:List[List[Value]], target:List[int]) -> Value:
        assert len(input) == len(target)
        return -sum(self.logsoftmax(yi)[ti] for yi,ti in zip(input, target)) / len(input)

    def __repr__(self) -> str:
        return "CrossEntropyLoss"