# rules:
# 1. define simple interfaces in ABCs
# 2. implementations require interfaces
# 3. users require implementations
# where require = inherit from

from functools import wraps
import abc
import six

from ..utils import todo, todocument


@todocument
@todo('change run_forward arguments to feed_dict for more user control')
class Forwardable(six.with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def inputs(self):
        """Returns a tensor or list of input tensors which can be used as an
        argument to the `forward` method.
        """
        return NotImplemented

    @abc.abstractproperty
    def placeholders(self):
        """Property to access tensors which can serve as placeholders to feed
        the model in the `run_forward` method.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def forward(self, inputs):
        """Returns the output tensor of the model.
        
        Arguments:
            inp: tensor or list of input tensors.
        """
        return NotImplemented

    @abc.abstractmethod
    def run_forward(self, inp):
        return NotImplemented

    @abc.abstractmethod
    def build_inference_graph(self):
        return NotImplemented
    
    @abc.abstractproperty
    def forward_op(self):
        return NotImplemented


class Lossable(six.with_metaclass(abc.ABCMeta, object)):
    
    @abc.abstractmethod
    def loss(self, outputs, targets):
        """Returns the loss of the model.
        
        Arguments:
            outputs: tensor or list of the model's output tensors.
            targets: tensor or list of the model's target tensors.
        """
        return NotImplemented


@todocument
class Trainable(six.with_metaclass(abc.ABCMeta, object)):
    
    @abc.abstractmethod
    def train_data(self):
        """Returns the input to the model for training purposes.
        """
        return NotImplemented

    @abc.abstractmethod
    def train(self, loss):
        """Returns an operation to update the model's parameters.
        
        Arguments:
            loss: tensor the method must minimize.
        """
        return NotImplemented

    @abc.abstractmethod
    def run_train(self):
        return NotImplemented

    @abc.abstractmethod
    def build_train_graph(self):
        return NotImplemented
    
    @abc.abstractproperty
    def train_op(self):
        return NotImplemented


@todocument
class Testable(six.with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def test_data(self):
        """Returns the input to the model for testing purposes.
        """
        return NotImplemented

    @abc.abstractmethod
    def test(self, outputs, targets):
        """Returns an operation to evaluate the model.
        
        Arguments:
            outputs: tensor or list of the model's output tensors.
            targets: tensor or list of the model's target tensors.
        """
        return NotImplemented

    @abc.abstractmethod
    def run_test(self):
        return NotImplemented

    @abc.abstractmethod
    def build_test_graph(self):
        return NotImplemented
    
    @abc.abstractproperty
    def test_op(self):
        return NotImplemented


class Saveable(six.with_metaclass(abc.ABCMeta, object)):
    
    @abc.abstractproperty
    def save_path(self):
        """Specifies path to which variables should be saved."""
        return NotImplemented

    @abc.abstractmethod
    def save(self):
        """Save the variables of a graph.
        """
        return NotImplemented

    @abc.abstractmethod
    def restore(self):
        """Restore the variables of a graph.
        """
        return NotImplemented


class Exportable(six.with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def export(self, path=None):
        """Save the inferable graph of the model in an exportable format.

        Arguments:
            path: directory in which to export.
        """
        return NotImplemented


@todocument
class Traceable(six.with_metaclass(abc.ABCMeta, object)):

    @abc.abstractproperty
    def options(self):
        return NotImplemented
    
    @abc.abstractproperty
    def run_metadata(self):
        return NotImplemented


@todocument
@todo('add abstractmethods to further hide session & graph object from user')
class TFInfo(six.with_metaclass(abc.ABCMeta, object)):

    @abc.abstractproperty
    def graph(self):
        return NotImplemented

    @abc.abstractproperty
    def session(self):
        return NotImplemented

    @abc.abstractmethod
    def finalize(self):
        return NotImplemented

    @abc.abstractproperty
    def finalized(self):
        return NotImplemented