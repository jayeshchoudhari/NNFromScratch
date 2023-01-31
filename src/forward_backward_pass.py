import math
import numpy as np


class Value:
    
    def __init__(self, data, _children=(), _op='', label = ''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _local_backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _local_backward
        
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _local_backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _local_backward
        
        return out
    
    def exp(self, multiplier):
        return math.exp(multiplier * self.data)

    def tanh(self):
        x = self.data
        t = (self.exp(2)  - 1)/(self.exp(2)  + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _local_backward():
            self.grad += (1 - (t**2)) * out.grad
            
        out._backward = _local_backward
        
        return out

    def backward(self):
        topo = []
        visited = set()
        # labels = []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                # labels.append(v.label)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            # print(node.label)
            node._backward()


if __name__ == '__main__':
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias
    b = Value(6.88137358701954, label='b')


    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1+x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label= 'o'

    o.backward()
