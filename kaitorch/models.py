import kaitorch

from kaitorch import activations as A
from kaitorch import functional as F

from kaitorch.core import Module, Scalar
from kaitorch.layers import Dropout
from kaitorch.graph import plot_model
from kaitorch.utils import ffill, unwrap, wrap
from kaitorch.optimizers import Optimizer

from tqdm import tqdm


class Sequential(Module):

    def __init__(self, layers=None):
        self.built = False
        self.compiled = False

        self.layers = layers if layers else []
        self.layer_sizes = [layer.nouts for layer in self.layers] if self.layers else []

    def __call__(self, x, train):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer(x, train)
            else:
                x = layer(x)
        return unwrap(x)

    def __repr__(self):
        print([layer.parameters() for layer in self.layers])
        return '\n'.join(str(layer) for layer in self.layers)

    def summary(self):
        print("_" * 115)
        print(
            "Layer (params)"
            + " " * 59
            + "Output Shape"
            + " " * 5
            + "Params = Weights + Biases"
        )
        print("=" * 115)
        for layer_num, layer in enumerate(self.layers):
            l_name = layer.__repr__()
            l_output = f"(None, {layer.nouts})"
            l_params = len(layer.parameters())
            l_w = l_params - layer.nouts if l_params > 0 else 0
            l_b = layer.nouts if l_params > 0 else 0

            print(f"{l_name:<73}{l_output:<17}{l_params:<9}{l_w:<10}{l_b:<6}")
            if layer_num != (len(self.layers) - 1):
                print("_" * 115)
        print("=" * 115)
        print(
            f"Total Params: {sum([len(layer.parameters()) for layer in self.layers])}"
        )
        print("_" * 115)

    def add(self, layer):

        self.layers.append(layer)
        self.layer_sizes.append(layer.nouts)

    def build(self, input_size):

        if self.built:
            return

        self.layer_sizes.insert(0, input_size)
        self.layer_sizes = ffill(self.layer_sizes)

        for idx, layer in enumerate(self.layers):
            layer.__build__(self.layer_sizes[idx])

        self.built = True

    def plot(self, filename=None):

        if not self.built:
            raise Exception(
                '[Model Not Built] - Use Sequential.build(input_size) to build model'
            )
        empty_input = self.__call__([0]*self.layer_sizes[0], train=False)
        return plot_model(empty_input, filename=filename)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def compile(self, optimizer, loss):

        def set_optimizer(optimizer):
            if isinstance(optimizer, str):
                if optimizer in kaitorch.optimizers.__all__:
                    self.optimizer = getattr(kaitorch.optimizers, optimizer)()
                else:
                    raise Exception(
                        f'[Undefined Optimizer] - Optimizer "{optimizer}" has not been implemented'
                    )
            elif isinstance(optimizer, Optimizer):
                self.optimizer = optimizer
            else:
                raise Exception(
                    f'[Undefined Optimizer] - Object passed was not an Optimizer'
                )

        def set_loss(loss):
            if isinstance(loss, str):
                if loss in kaitorch.losses.__all__:
                    self.loss = getattr(kaitorch.losses, loss)()
                else:
                    raise Exception(
                        f'[Undefined Loss Function] - Loss Function "{loss}" has not been implemented'
                    )
            else:
                self.loss = loss

        if not self.compiled:
            if optimizer and loss:
                set_optimizer(optimizer)
                set_loss(loss)
                self.compiled = True
            else:
                raise Exception(
                    '[Unable to Compile] - Optimizer and Loss Function must be specified'
                )

    def step(self, **optimizer_params):

        if not self.compiled:
            raise Exception('[Missing Optimizer] - Model has not been compiled')

        for p in self.parameters():
            self.optimizer(p)

    def run(self, x, y=None, epoch=1, epochs=1, train=False):

        postfix_type = 'Train' if train is True else ''

        tqdm_x = tqdm(
            x,
            ncols=160,
            desc=f"Epoch {epoch:>3}/{epochs}", 
            postfix='',
            bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        )

        y_pred = []

        for x_record in tqdm_x:
            y_pred.append(self.__call__(x_record, train=train))
            if y:
                run_loss = self.loss(y[:len(y_pred)], y_pred)
                tqdm_x.set_postfix_str(f"{postfix_type} Loss: {run_loss.data:.4f}")
            else:
                run_loss = None
                tqdm_x.set_postfix_str(f"{postfix_type}")

        if train:
            self.zero_grad()
            run_loss.backward()
            self.step()

        return y_pred, run_loss

    def fit(self, x, y, epochs=1):

        x = wrap(x)
        self.build(len(x[0]))

        history = {'loss': []}

        for epoch in range(1, epochs+1):

            y_pred, run_loss = self.run(x, y, epoch, epochs, train=True)
            history['loss'].append(run_loss.data)

        return history

    def evaluate(self, x, y):

        x = wrap(x)
        self.build(len(x[0]))

        evaluation = {'loss': []}

        y_pred, run_loss = self.run(x, y)
        evaluation['loss'].append(run_loss.data)

        return evaluation

    def predict(self, x, as_scalar=False):

        x = wrap(x)
        self.build(len(x[0]))

        y_pred, run_loss = self.run(x)

        if as_scalar:
            return [y for y in y_pred]
        else:
            if isinstance(y_pred[0], Scalar):
                return [y.data for y in y_pred]
            elif isinstance(y_pred[0][0], Scalar):
                return [[y.data for y in row] for row in y_pred]
