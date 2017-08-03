import tensorflow as tf


from . import interfaces
from ..utils import totest, todocument


@totest
@todocument
class TrainGraph(
    interfaces.Forwardable,
    interfaces.Trainable,
    interfaces.Lossable,
    interfaces.TFInfo,
):

    def build_train_graph(self):
        with self.graph.as_default():
            inputs, targets = self.train_data()
            outputs = self.forward(inputs)
            loss = self.loss(outputs, targets)
            self.train_op = self.train(loss)
            return self.train_op

    @property
    def train_op(self):
        return getattr(self, '_train_op', None)
    
    @train_op.setter
    def train_op(self, val):
        self._train_op = val

    def run_train(self):
        assert self.train_op is not None,\
            'call build_train_graph before running train graph'
        return self.run(self.train_op)

    def training(self, limit=None, step=-1):
        """Yields the return of run_train `limit` - `step` times.
        `limit` defaults to None, used to run indefinitely.
        """
        for step in iter(lambda: step + 1, limit):
            yield self.run_train()
