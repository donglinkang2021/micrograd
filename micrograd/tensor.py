import math

class Value:
    def __init__(self, data, _children=(), _op='', label:str=None):
        self.data = data
        self.grad = 0.0
        # for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        # for graphviz / debugging / etc
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1.0 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Value(1.0 / (1.0 + math.exp(-self.data)), (self,), 'sigmoid')
        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0/self.data) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def topo(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        return topo
    
    def backward(self):
        topo = self.topo()
        # go one variable at a time 
        # and apply the chain rule to get its gradient
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
