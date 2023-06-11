import random
from kaitorch.utils import unwrap, wrap
from kaitorch.core import Scalar, Module
from kaitorch.initializers import Initializer
from kaitorch import activations as A
from kaitorch import initializers as I


class Dense(Module):

    class Node:

        def __init__(self, nin, nout, activation, initializer):

            self.w = [Scalar(initializer(nin, nout)) for _ in range(nin)]
            self.b = Scalar(initializer(nin, nout))
            self.a = activation

        def __call__(self, x):
            x = wrap(x)
            signal = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
            if self.a and self.a != 'softmax':
                signal = signal.activation(self.a)
            return signal

        def parameters(self):
            return self.w + [self.b]

    def __init__(self, nouts, activation=None, initializer='glorot_uniform'):
        self.nins = None
        self.nouts = nouts
        self.nodes = None
        self.activation = activation
        self.initializer = self.get_initializer(initializer)

    def get_initializer(self, initializer):

        if isinstance(initializer, str):
            if initializer in I.__all__:
                return getattr(I, initializer)()
            else:
                raise Exception(
                    f'[Undefined Initializer] - Initializer "{initializer}" has not been implemented'
                )
        elif isinstance(initializer, Initializer):
            return initializer
        else:
            raise Exception(
                '[Undefined Initializer] - Object passed was not a str or Initializer'
            )

    def __repr__(self):
        repr_str = f'Dense(units={self.nouts}'
        if self.activation is not None:
            repr_str += f', activation={self.activation}'
        if self.initializer is not None:
            repr_str += f', initializer={self.initializer}'
        return repr_str + ')'

    def __build__(self, nins):
        self.nins = nins
        self.nodes = [
            self.Node(self.nins, self.nouts, self.activation, self.initializer)
            for _ in range(self.nouts)
        ]

    def __call__(self, x):
        outs = [n(x) for n in self.nodes]
        if self.activation == 'softmax':
            outs = getattr(A, self.activation)(outs)
        return unwrap(outs)

    def parameters(self):
        return [p for node in self.nodes for p in node.parameters()]


class Dropout(Module):

    class Node:

        def __init__(self, dropout_rate: float):
            self.p = 1 - dropout_rate

        def __call__(self, x, train):
            if train:
                return x * (1/self.p) if random.random() <= self.p else 0
            else:
                return x

        def parameters(self):
            return []

    def __init__(self, dropout_rate: float = 0.5):

        self.nins = None
        self.nouts = None
        self.nodes = None
        self.q = dropout_rate

        if self.q < 0 or self.q > 1:
            raise ValueError("p must be a probability")

    def __repr__(self):
        return f'Dropout(dropout_rate={self.q})'

    def __build__(self, nins):
        self.nins = nins
        self.nouts = nins
        self.nodes = [self.Node(self.q) for _ in range(self.nins)]

    def __call__(self, x, train):
        outs = [n(xi, train) for n, xi in zip(self.nodes, x)]
        return unwrap(outs)

    def parameters(self):
        return [p for node in self.nodes for p in node.parameters()]
