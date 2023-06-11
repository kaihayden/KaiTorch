import math
from kaitorch.core import Scalar

__all__ = ['SGD', 'Momentum', 'Nesterov', 'Adagrad', 'RMSprop', 'Adam']


class Optimizer:
    pass


# Stochastic Gradient Descent
class SGD(Optimizer):

    def __init__(self, lr=0.01, decay_rate=1.0):
        self.lr = lr
        self.decay_rate = decay_rate

    def __call__(self, p: Scalar):

        # θ'   = θ      - (α       * ▽f(θ) )
        p.data = p.data - (self.lr * p.grad)

        self.lr *= self.decay_rate

    def __repr__(self):
        return f'SGD(lr={self.lr})'


# Stochastic Gradient Descent with Momentum
class Momentum(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9, decay_rate=1.0):
        self.lr = lr
        self.momentum = momentum
        self.decay_rate = decay_rate

    def __call__(self, p: Scalar):

        if not hasattr(p, 'm'):
            p.m = 0.0

        # m'= η             * m   + (1 - η            ) * ▽f(θ)
        p.m = self.momentum * p.m + (1 - self.momentum) * p.grad

        # θ'   = θ      - α       * m'
        p.data = p.data - self.lr * p.m

        self.lr *= self.decay_rate

    def __repr__(self):
        return f'Momentum(lr={self.lr}, Momentum={self.momentum})'


# Stochastic Gradient Descent with Nesterov Accelerated Gradient
class Nesterov(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9, decay_rate=1.0):
        self.lr = lr
        self.momentum = momentum
        self.decay_rate = decay_rate

    def __call__(self, p: Scalar):

        if not hasattr(p, 'm'):
            p.m = 0.0

        # m'= (η             * m  ) - (α       * ▽f(θ) )
        p.m = (self.momentum * p.m) - (self.lr * p.grad)

        # θ'   = θ      + (η             * m' ) - (α       * ▽f(θ) )
        p.data = p.data + (self.momentum * p.m) - (self.lr * p.grad)

        self.lr *= self.decay_rate

    def __repr__(self):
        return f'Nesterov(lr={self.lr}, Momentum={self.momentum})'


# Adaptive Gradient Algorithm
class Adagrad(Optimizer):

    def __init__(self, lr=0.01, epsilon=1e-8, decay_rate=1.0):
        self.lr = lr
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def __call__(self, p: Scalar):

        if not hasattr(p, 'v'):
            p.v = 0.0

        # v'= v   + ▽f(θ)^2
        p.v = p.v + p.grad ** 2

        # θ'   = θ      - α       * ▽f(θ)  / (        √ v'   + ε           )
        p.data = p.data - self.lr * p.grad / (math.sqrt(p.v) + self.epsilon)

        self.lr *= self.decay_rate

    def __repr__(self):
        return f'Adagrad(lr={self.lr})'


# Root Mean Square Propogation
class RMSprop(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8, decay_rate=1.0):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def __call__(self, p: Scalar):

        if not hasattr(p, 'v'):
            p.v = 0.0

        # v'= ρ        * v   + (1 - ρ       ) * ▽f(θ)^2
        p.v = self.rho * p.v + (1 - self.rho) * p.grad ** 2

        # θ'   = θ      - α       * ▽f(θ)  / (        √ v'   + ε)
        p.data = p.data - self.lr * p.grad / (math.sqrt(p.v) + self.epsilon)

        self.lr *= self.decay_rate

    def __repr__(self):
        return f'RMSprop(lr={self.lr}, rho={self.rho})'


# Adaptive Moment Estimation
class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def __call__(self, p):

        if not hasattr(p, 'm'):
            p.m = 0.0
        if not hasattr(p, 'v'):
            p.v = 0.0

        # First and Second Moment Estimation

        # m'= β1         * m   + (1 - β1        ) * ▽f(θ)
        p.m = self.beta1 * p.m + (1 - self.beta1) * p.grad
        # v'= β1         * v   + (1 - β2        ) * ▽f(θ)^2
        p.v = self.beta2 * p.v + (1 - self.beta2) * p.grad ** 2

        # Bias Correction

        # m^  = m'  / (1 - β1)
        m_hat = p.m / (1 - self.beta1)
        # v^  = v'  / (1 - β2)
        v_hat = p.v / (1 - self.beta2)

        # Parameter Update

        # θ'   = θ      - α       * m^    / (        √ v^   ) + ε           )
        p.data = p.data - self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)

        self.lr *= self.decay_rate

    def __repr__(self):
        return f'Adam(lr={self.lr}, β1={self.beta1}, β2={self.beta2})'
