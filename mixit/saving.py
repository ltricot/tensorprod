from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import builder, tag_constants, loader
import tensorflow as tf

from functools import wraps

from . import interfaces
from . import testing
from . import diverse
from ..utils import totest, todocument, todo


@todocument
@todo('implement more specific savers -- only model variables, for example')
class Saved(
    interfaces.TFInfo,
    interfaces.Saveable,
):

    def initialize_saver(self):  # called by user
        with self.graph.as_default():
            variables = self.graph.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(variables)

    @property
    def saver(self):
        if not hasattr(self, '_saver'):
            raise AttributeError('saver isn\'t set')
        return self._saver

    @saver.setter
    def saver(self, val):
        assert isinstance(val, tf.train.Saver)
        self._saver = val

    @wraps(tf.train.Saver.save)  # inherit documentation
    def save(self, sess=None, save_path=None, **kws):
        sess = sess or self.session
        save_path = save_path = self.save_path
        self.saver.save(sess=sess, save_path=save_path, **kws)

    @wraps(tf.train.Saver.restore)  # inherit documentation
    def restore(self, sess=None, save_path=None, **kws):
        sess = sess or self.session
        save_path = save_path or self.save_path
        self.saver.restore(sess=sess, save_path=save_path, **kws)


@totest
@todocument
@todo('implement load method -- map operations to properties ; implement freeze_graph based exporter')
class SavedModelExported(
    interfaces.TFInfo,
    interfaces.Exportable,
    interfaces.Forwardable,
):

    SERVING = tag_constants.SERVING  # == 'serve'
    TRAINING = tag_constants.TRAINING  # == 'train'
    TESTING = 'test'

    # TODO take signature as argument and let another mixin create it
    # or the user.
    def export(self, path):
        # assumes self.graph contains *all* necessary variables
        # assert there are variables to save
        assert self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # save current graph -- assumed to be for training & testing
        saved_model_builder = builder.SavedModelBuilder(path)
        saved_model_builder.add_meta_graph_and_variables(
            session=self.session,
            tags=[self.TRAINING, self.TESTING],
        )

        # build inference ops inn new graph
        self.graph = Graph()
        with self.graph.as_default():
            inputs = self.inputs()
            outputs = self.forward(inputs)

            # build prediction signature def for tf serving
            if isinstance(inputs, tf.Tensor):
                inputs = (inputs,)
            if isinstance(outputs, tf.Tensor):
                outputs = (outputs,)

            sigdef = predict_signature_def(
                inputs={op.name for op in inputs},
                outputs={op.name for op in outputs},
            )

            # add inference meta graph
            saved_model_builder.add_meta_graph(
                tags=[self.SERVING],
                signature_def_map={
                    'predict': sigdef,
                }
            )

            saved_model_builder.save()

    def load(self, tags, path):
        assert self.session is not None
        loader.load(self.session, tags, path)
