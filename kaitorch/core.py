import math

__all__ = ['Scalar', 'Module']


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Scalar:

    def __init__(self, data, _in=(), _op=''):
        self.data = data
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = set(_in)
        self._op = _op

    def __repr__(self):
        return f'Scalar(data={self.data})'

    def __add__(a, b):

        a = a if isinstance(a, Scalar) else Scalar(a)
        b = b if isinstance(b, Scalar) else Scalar(b)

        # Calculation: y = a + b
        def _forward():
            _a = a.data
            _b = b.data
            _y = _a + _b
            return Scalar(_y, _in=(a, b), _op='+')

        y = _forward()

        # Derivative: dy/da = 1
        # Chain Rule: dL/da = dL/dy * dy/da
        #                   = dL/dy
        def _backward():
            a.grad += y.grad
            b.grad += y.grad

        y._backward = _backward

        return y

    def __radd__(a, b):
        # b + a = a + b
        return a.__add__(b)

    def __mul__(a, b):

        a = a if isinstance(a, Scalar) else Scalar(a)
        b = b if isinstance(b, Scalar) else Scalar(b)

        # Calculation: y = a * b
        def _forward():
            _a = a.data
            _b = b.data
            _y = _a * _b
            return Scalar(_y, _in=(a, b), _op='*')

        y = _forward()

        # Derivative: dy/da = b
        # Chain Rule: dL/da = dL/dy * dy/da
        #                   = dL/dy * b
        def _backward():
            a.grad += y.grad * b.data
            b.grad += y.grad * a.data

        y._backward = _backward

        return y

    def __rmul__(a, b):
        # b * a = a * b
        return a.__mul__(b)

    def __neg__(a):
        # -a = a * -1
        return a.__mul__(-1)

    def __sub__(a, b):
        # a - b = a + (b * -1)
        return a.__add__(b.__neg__())

    def __rsub__(a, b):
        # b - a = (a * -1) + b
        return (a.__neg__()).__add__(b)

    def __pow__(a, b):

        assert isinstance(b, (int, float)), "Exponent is not int/float"

        # Calculation: y = a ** b
        def _forward():
            _a = a.data
            _y = (_a + 1e-8) ** b  # don't divide by 0 :)
            return Scalar(_y, _in=(a,), _op=f'**{b}')

        y = _forward()

        # Derivative: dy/da = b * (a ** (b-1))
        # Chain Rule: dL/da = dL/dy * dy/da
        #                   = dL/dy * b * (a ** (b-1))
        def _backward():
            a.grad += y.grad * (b * a.data ** (b - 1))

        y._backward = _backward

        return y

    def __truediv__(a, b):
        # a / b = a * (b ** -1)
        return a.__mul__((b + 1e-8).__pow__(-1))

    def __rtruediv__(a, b):
        # b / a = b * (a ** -1)
        return b.__mul__((a + 1e-8).__pow__(-1))

    def exp(a):

        # Calculation: y = e ** a
        def _forward():
            _a = a.data
            _y = math.exp(_a)
            return Scalar(_y, _in=(a, ), _op='exp')

        y = _forward()

        # Derivative: dy/da = y
        # Chain Rule: dL/da = dL/dy * dy/da
        #                   = dL/dy * y
        def _backward():
            a.grad += y.grad * y.data

        y._backward = _backward

        return y

    def log(a):

        # Calculation: y = ln(a)
        def _forward():
            _a = a.data
            _y = math.log(_a + 1e-8)
            return Scalar(_y, _in=(a, ), _op='ln')

        y = _forward()

        # Derivative: dy/da = 1/a * a'
        # Chain Rule: dL/da = dL/dy * dy/da * a'
        #                   = dL/dy * 1/a * a'
        def _backward():
            a.grad += y.grad * ((a.data + 1e-8).__pow__(-1))

        y._backward = _backward

        return y

    def activation(self, activation):

        import kaitorch.activations as A

        available = A.__all__

        if isinstance(activation, str) and activation in available:
            return getattr(A, activation)()(self)

        elif isinstance(activation, A.Activation):
            return activation(self)

        else:
            raise Exception(f'Activation {activation} not in {available}')

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
