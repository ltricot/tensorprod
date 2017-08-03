import tensorflow as tf


from . import interfaces
from ..utils import totest, todocument


@totest
@todocument
class TestGraph(
    interfaces.Forwardable,
    interfaces.Testable,
    interfaces.TFInfo,
):

    def build_test_graph(self):
        with self.graph.as_default():
            inputs, targets = self.test_data()
            outputs = self.forward(inputs)
            self.test_op = self.test(outputs, targets)
            return self.test_op

    @property
    def test_op(self):
        return getattr(self, '_test_op', None)

    @test_op.setter
    def test_op(self, val):
        self._test_op = val

    def run_test(self):
        assert self.test_op is not None,\
            'call build_test_graph before running test graph'
        return self.run(self.test_op)


@totest
@todocument
class InferGraph(
    interfaces.Forwardable,
    interfaces.TFInfo,
):

    def build_inference_graph(self):
        with self.graph.as_default():
            self.placeholders = self.inputs()
            self.forward_op = self.forward(self.placeholders)
            return self.forward_op
    
    @property
    def placeholders(self):
        return getattr(self, '_placeholders', None)

    @placeholders.setter
    def placeholders(self, val):
        # TODO type check
        self._placeholders = val

    @property
    def forward_op(self):
        return getattr(self, '_forward_op', None)

    @forward_op.setter
    def forward_op(self, val):
        self._forward_op = val

    def run_forward(self, inputs):
        # run forward op using feed_dict
        return self.run(
            self.forward_op,
            feed_dict=dict(zip([self.placeholders], [inputs])),
        )
