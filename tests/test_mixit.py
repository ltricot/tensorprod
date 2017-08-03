# rules:
# 1. require user contract
# 2. require the mixin you want to test
# 3. require its dependencies
# 4. write doctest
# where require = inherit from

from tensorprod.mixit import diverse, training, testing, saving
import tensorflow as tf

from .mixtest_ import UserContract, INPUT_DIM
from ..utils import todo


@todo('add finalize call')
class InferGraph(
    UserContract,
    testing.InferGraph,
    diverse.DefaultTFInfo,
):
    """Test the `InferGraph` mixin.

    >>> model = InferGraph()
    >>> model.initialize()
    >>> model.build_inference_graph()
    <tf.Tensor 'forward/outputs:0' shape=(?, 10) dtype=float32>
    >>> model.forward_op
    <tf.Tensor 'forward/outputs:0' shape=(?, 10) dtype=float32>
    >>> model.initialize_variables()
    >>>
    >>> import numpy as np
    >>> inputs = np.random.normal(size=(30, INPUT_DIM))
    >>> outputs = model.run_forward(inputs)
    >>> outputs.shape
    (30, 10)
    """
    pass


@todo('add finalize call')
class TrainGraph(
    UserContract,
    training.TrainGraph,
    testing.InferGraph,
    diverse.DefaultTFInfo,
):
    """Test the `TrainGraph` mixin.

    >>> model = TrainGraph()
    >>> model.initialize()  # TFInfo creates graph & session here
    >>> model.build_train_graph()
    <tf.Operation 'optimize/GradientDescent' type=NoOp>
    >>> model.train_op
    <tf.Operation 'optimize/GradientDescent' type=NoOp>
    >>> model.initialize_variables()
    >>> model.run_train()
    >>> for _ in model.training(limit=5):
    ...     pass
    ...
    """
    pass


class Saved(
    UserContract,
    saving.Saved,
    diverse.DefaultTFInfo,
    testing.InferGraph,  # something which creates variables
):
    """Test the `Saver` mixin.

    >>> from tempfile import mkstemp
    >>> from os import remove
    >>> 
    >>> # `open` can take a file descriptor or a path
    ... f, path = mkstemp()
    >>> 
    >>> # initialize a model
    ... model = Saved(save_path=path)
    >>> model.initialize()
    >>> model.build_inference_graph()
    <tf.Tensor 'forward/outputs:0' shape=(?, 10) dtype=float32>
    >>> model.initialize_saver()
    >>> model.initialize_variables()
    >>> 
    >>> # fetch a random variable
    ... var1 = model.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]
    >>> var1 = model.run(var1)  # grab value
    >>> 
    >>> model.save()
    >>> del model
    >>> 
    >>> model = Saved(save_path=path)
    >>> model.initialize()
    >>> model.build_inference_graph()
    <tf.Tensor 'forward/outputs:0' shape=(?, 10) dtype=float32>
    >>> model.initialize_saver()
    >>> model.restore()
    >>> 
    >>> var2 = model.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]
    >>> var2 = model.run(var2)
    >>> 
    >>> (var1 == var2).all()
    True
    >>> remove(path)
    """
    def __init__(self, save_path):
        self._save_path = save_path

    @property
    def save_path(self):
        return self._save_path


@todo('add finalize call')
class FullTrace(
    UserContract,
    diverse.FullTrace,
    training.TrainGraph,
    testing.InferGraph,
):
    """Test the `FullTrace` mixin.

    >>> from tempfile import mkstemp
    >>> from os import remove
    >>>
    >>> model = FullTrace()
    >>> model.initialize()
    >>> model.build_train_graph()
    <tf.Operation 'optimize/GradientDescent' type=NoOp>
    >>> model.initialize_variables()
    >>> _ = model.run_train()  # discard loss -- unstable over runs
    >>>
    >>> # `open` can take a file descriptor or a path
    ... f, path = mkstemp()
    >>> model.chrome_trace(f)  # you can now open `path` with chrome://tracing
    >>> remove(path)
    """
    pass

if __name__ == '__main__':
    import doctest
    doctest.testmod()
