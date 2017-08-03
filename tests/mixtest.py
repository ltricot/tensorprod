from __future__ import print_function

from tensorprod.mixit import interfaces

from collections import defaultdict


class Forwardable(interfaces.Forwardable):
    """Implements the `Forwardable` interface, to be inherited by test classes
    as a mock dependency.

    >>> model = Forwardable()
    >>> model.build_inference_graph()
    >>> feed = {model.placeholders: 'fed_input'}
    >>> model.run_forward(feed)
    ran forward(fed_input)
    """
    def inputs(self):
        return '{placeholders}'

    @property
    def placeholders(self):
        if not hasattr(self, '_placeholders'):
            raise AttributeError('placeholders not set')
        return self._placeholders

    @placeholders.setter
    def placeholders(self, val):
        self._placeholders = val.strip('{}')

    def forward(self, inputs):
        return 'forward(' + inputs + ')'

    def run_forward(self, feed_dict):
        print('ran', self.forward_op.format(**feed_dict))

    def build_inference_graph(self):
        inputs = self.inputs()
        self.placeholders = inputs
        self.forward_op = self.forward(inputs)

    @property
    def forward_op(self):
        if not hasattr(self, '_forward_op'):
            raise AttributeError('forward_op not set')
        return self._forward_op

    @forward_op.setter
    def forward_op(self, val):
        self._forward_op = val


class Lossable(interfaces.Lossable):
    """Implements the `Lossable` interface, to be inherited by test classes as a
    mock dependency.

    >>> lossable = Lossable()
    >>> lossable.loss('outputs', 'targets')
    'loss(outputs, targets)'
    """
    def loss(self, outputs, targets):
        return 'loss(' + outputs + ', ' + targets + ')'


class Trainable(
    interfaces.Trainable,
    interfaces.Lossable,
    interfaces.Forwardable,
):
    """Implements the `Trainable` interface, to be inherited by test classes as
    a mock dependency.

    >>> class Model(Trainable, Forwardable, Lossable):
    ...     pass
    ...
    >>> model = Model()
    >>> model.build_train_graph()
    >>> model.run_train()
    ran train(loss(forward(train_input), train_target))
    >>>
    >>> model.train_op
    'train(loss(forward(train_input), train_target))'
    """
    def train_data(self):
        return 'train_input', 'train_target'

    def train(self, loss):
        return 'train(' + loss + ')'

    def run_train(self):
        print('ran ' + self.train_op)

    def build_train_graph(self):
        inputs, targets = self.train_data()
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.train_op = self.train(loss)

    @property
    def train_op(self):
        if not hasattr(self, '_train_op'):
            raise AttributeError('train_op not set')
        return self._train_op

    @train_op.setter
    def train_op(self, val):
        self._train_op = val


class Testable(
    interfaces.Testable,
    interfaces.Forwardable,
):
    """Implements the `Testable` interface, to be inherited by test classes as a
    mock dependency.

    >>> class Model(Testable, Forwardable):
    ...     pass
    ...
    >>> model = Model()
    >>> model.build_test_graph()
    >>> model.run_test()
    ran test(forward(test_input), test_target)
    >>>
    >>> model.test_op
    'test(forward(test_input), test_target)'
    """
    def test_data(self):
        return 'test_input', 'test_target'

    def test(self, outputs, targets):
        return 'test(' + outputs + ', ' + targets + ')'

    def run_test(self):
        print('ran ' + self.test_op)

    def build_test_graph(self):
        inputs, targets = self.test_data()
        outputs = self.forward(inputs)
        self.test_op = self.test(outputs, targets)

    @property
    def test_op(self):
        if not hasattr(self, '_test_op'):
            raise AttributeError('test_op not set')
        return self._test_op

    @test_op.setter
    def test_op(self, val):
        self._test_op = val


class Saveable(
    interfaces.Saveable,
):

    @property
    def save_path(self):
        return 'save_path'

    def save(self):
        print('saved to', self.save_path)

    def restore(self):
        print('restored from', self.save_path)


class TestExport(
    interfaces.Exportable,
):

    def export(self, path=None):
        path = path or 'export_path'
        print('exported to {path}'.format(path=path))


class TFInfo(
    interfaces.TFInfo,
):

    class Session:
        """Mock tf.Session to avoid making tensorflow calls during testing."""
        def run(self, ops, **kws):
            print('ran', ops, 'with options', kws)

        def eval(self, tensors, **kws):
            print('evaluated', tensors, 'with options', kws)

        @property
        def graph(self):
            if not hasattr(self, '_graph'):
                self._graph = TFInfo.Graph()
            return self._graph

    class Graph:
        """Mock tf.Graph to avoid making tensorflow calls during testing. Must
        emulate collection behavior, because it may be useful to other mixins.
        """
        def add_to_collection(self, name, value):
            if not hasattr(self, '_collections'):
                self._collections = defaultdict(list)
            self._collections[name].append(value)

        def get_collection(self, name, scope=None):
            return getattr(self, _collections, {}).get(name, []).copy()

    @property
    def graph(self):
        return self._session.graph

    @property
    def session(self):
        if not hasattr(self, '_session'):
            self._session = TFInfo.Session()
        return self._session

    def finalize(self):
        self.finalized = True

    @property
    def finalized(self):
        return getattr(self, '_finalized', False)

    @finalized.setter
    def finalized(self, val):
        self._finalized = val


if __name__ == '__main__':
    import doctest
    doctest.testmod()
