import tensorflow as tf
import numpy as np

import numbers


def _feature(arr):
    """Returns a tf.train.Feature as defined in
    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/example/feature.proto
    from a 1-dimensional array.

    Arguments:
        arr: 1-dimensional array, of dtype number or bytes. Numpy convertible.

    >>> _feature([0, 1, 2, 3])
    int64_list {
      value: 0
      value: 1
      value: 2
      value: 3
    }
    <BLANKLINE>
    """
    arr = np.asarray(arr)
    assert arr.ndim == 1
    for abc, dtype in [
        (numbers.Integral, 'int64'),
        (numbers.Real, 'float'),
        (bytes, 'bytes'),
    ]:
        if issubclass(arr.dtype.type, abc):
            break
    else:
        raise ValueError('neither int, float or bytes')

    list_ = {
        'int64': tf.train.Int64List,
        'bytes': tf.train.BytesList,
        'float': tf.train.FloatList,
    }[dtype](value=arr)
    kws = {'%s_list' % dtype: list_}
    return tf.train.Feature(**kws)

def _feature_list(arr):
    """Returns a tf.train.FeatureList as defined in
    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/example/feature.proto
    from a 2 dimensional array.

    Arguments:
        arr: 2-dimensional array, of dtype number or bytes. Numpy convertible.

    >>> _feature_list([[0, 1, 2, 3], [-1, 2, -2, 4]])
    feature {
      int64_list {
        value: 0
        value: 1
        value: 2
        value: 3
      }
    }
    feature {
      int64_list {
        value: -1
        value: 2
        value: -2
        value: 4
      }
    }
    <BLANKLINE>
    """
    arr = np.asarray(arr)
    assert arr.ndim == 2
    feature = [_feature(row) for row in arr]
    return tf.train.FeatureList(feature=feature)

def _features(dic):
    """Returns a tf.train.Features as defined in
    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/example/feature.proto
    from a dictionary mapping strings to 1-dimensional arrays.

    Arguments:
        dic: maps from strings to 1-dimensional arrays of dtype number or
        bytes. Numpy convertible.

    >>> _features({'color': [0, 1, 0, 0], 'size': [3.4]})
    feature {
      key: "color"
      value {
        int64_list {
          value: 0
          value: 1
          value: 0
          value: 0
        }
      }
    }
    feature {
      key: "size"
      value {
        float_list {
          value: 3.4
        }
      }
    }
    <BLANKLINE>
    """
    for key in dic:
        assert isinstance(key, str)
        dic[key] = _feature(dic[key])
    return tf.train.Features(feature=dic)

def _feature_lists(dic):
    """Returns a tf.train.FeatureLists as defined in
    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/example/feature.proto
    from a dictionary mapping strings to 2-dimensional arrays.

    Arguments:
        dic: maps from strings to 2-dimensional arrays of dtype number or
        bytes. Numpy convertible.

    >>> _feature_lists({
    ...     'word_embeddings': [[4., -1., 2.3, 0.], [2., 3.4, -1., -1.7]],
    ...     'word_tags': [[bytes('BIC', 'ascii')], [bytes('IBAN', 'ascii')]],
    ... })
    feature_list {
      key: "word_embeddings"
      value {
        feature {
          float_list {
            value: 4.0
            value: -1.0
            value: 2.3
            value: 0.0
          }
        }
        feature {
          float_list {
            value: 2.0
            value: 3.4
            value: -1.0
            value: -1.7
          }
        }
      }
    }
    feature_list {
      key: "word_tags"
      value {
        feature {
          bytes_list {
            value: "BIC"
          }
        }
        feature {
          bytes_list {
            value: "IBAN"
          }
        }
      }
    }
    <BLANKLINE>
    """
    for key in dic:
        assert isinstance(key, str)
        dic[key] = _feature_list(dic[key])
    return tf.train.FeatureLists(feature_list=dic)

def Example(dic):
    """Returns a tf.train.Example as defined in
    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/example/example.proto
    from a dictionary fititng the _features function.

    dic format:
    {
        'feature-1': [val, val, val],
        'feature-2': [val, val, val],
    }

    This example may be used to train a static input model.

    See _features for tests.
    """
    return tf.train.Example(features=_features(dic))

def SequenceExample(dic, ctx=None):
    """Returns a tf.train.SequenceExample as defined in
    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/core/example/example.proto
    from a dictionary fitting the _feature_lists function and another fitting
    the _features function.

    dic format:
    {
        'feature-1': [[val, val], [val, val]],
        'feature-2': [[val, val], [val, val]],
    }
    where the first axis is the sequential one -- time, for example.

    This sequential example may be used to train a sequential input model.

    See _features and _feature_lists for tests.
    """
    if ctx is None:
        ctx = dict()
    return tf.train.SequenceExample(
        feature_lists=_feature_lists(dic),
        context=_features(ctx),
    )


class TFRecordWriter:

    def __init__(self, pattern, n, i=-1):
        def writer(i=-1):
            for i in iter(lambda: i + 1, None):
                # change file & writer every records
                with tf.train.TFRecordWriter(pattern.format(i)) as tfr_writer:
                    for _ in range(n):
                        rec = yield
                        if rec is None: return
                        tfr_writer.write(rec)

        self._writer = writer(i=i)
        self._writer.send(None)  # prime coro

    def write(self, rec):
        self.writer.send(rec)

    def close(self):
        try:
            self._writer.send(None)
        except StopIteration:
            pass

    def __enter__(self):
        return self

    def __exit__(self, etype, eval, tb):
        self.close()
    
    def __del__(self):
        self.close()

if __name__ == '__main__':
    import doctest
    doctest.testmod()