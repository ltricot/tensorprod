from tensorflow.python.client import timeline
import tensorflow as tf

from functools import wraps

from . import interfaces
from ..utils import totest, todocument, todo


@totest
@todocument
@todo('finalize graph on `finalize` call')
class DefaultTFInfo(
    interfaces.TFInfo,
):
    """Manages a tensorflow graph and session.
    When accessed, the `session` property tries to return, in the following
    order:
        * the session set by the last access,
        * the tensorflow default session,
        * a new session, which is then set to be returned by the next call.

    The `graph` property returns `self.session.graph`.
    """

    @property
    def graph(self):
        return getattr(self, '_graph', None)

    @graph.setter
    def graph(self, val):
        assert isinstance(val, tf.Graph)
        self._graph = val
    
    @property
    def session(self):
        return getattr(self, '_session', None)

    @session.setter
    def session(self, val):
        assert isinstance(val, tf.Session)
        self._session = val

    def initialize(self, graph=None, session=None):
        self.graph = graph or tf.Graph()
        self.session = session or tf.Session(graph=self.graph)

    def initialize_variables(self, init=None):  # called by user
        with self.graph.as_default():
            init = init or tf.global_variables_initializer()
        return self.run(init)

    def finalize(self):
        self._finalized = True

    @property
    def finalized(self):
        return getattr(self, '_finalized', False)

    def run(self, *args, **kws):
        # proxy, encapsulates session
        assert self.session is not None
        return self.session.run(*args, **kws)


@totest
def reusevariables(method):
    """A marker to indicate this method should reuse the variables it creates
    across calls.
    """
    @wraps(method)
    def wrapper(*args, **kws):
        scope = _used.get(method, None)
        if scope is None:
            scope = tf.variable_scope(method.__name__, reuse=False)
        with scope:
            res = method(*args, **kwargs)
            scope.reuse_variables()
        return res
    return wrapper


@todocument
class Traced(
    interfaces.Traceable,
    DefaultTFInfo,
):

    def run(self, operation):
        # update run metadata as we run a train step
        return self.session.run(
            operation,
            options=self.options,
            run_metadata=self.run_metadata
        )

    def chrome_trace(self, path):
        # create a timeline object and write it to a json file
        t = timeline.Timeline(self.run_metadata.step_stats)
        chrome_trace = t.generate_chrome_trace_format()
        with open(path, 'w') as f:
            f.write(chrome_trace)


@todocument
@todo('mixin factory to choose trace level')
class FullTrace(
    Traced,
):
    """Use this instead of `DefaultTFInfo` if you want to profile your graph."""

    @property
    def options(self):
        return tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    
    @property
    def run_metadata(self):
        if not hasattr(self, '_run_metadata'):
            self._run_metadata = tf.RunMetadata()
        return self._run_metadata
