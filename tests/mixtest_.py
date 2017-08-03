# implement a coherent user contract inheritable by all mixin test classes
import tensorflow as tf

from ..utils import todo


# constants to define model architecture
INPUT_DIM = 100
OUTPUT_DIM = 10
LR = .01
MBS = 30


@todo('remove trainling underscore from module name')
class UserContract:
    """Simple classifer trained on random data."""

    def inputs(self):
        return tf.placeholder(tf.float32, [None, INPUT_DIM], name='inputs')

    def forward(self, inputs):
        with tf.variable_scope('forward'):
            shape = inputs.get_shape().as_list()

            weights = tf.get_variable('weights',
                shape=(shape[1], OUTPUT_DIM),
                initializer=tf.random_normal_initializer(dtype=tf.float32))
            biases = tf.get_variable('biases',
                shape=[OUTPUT_DIM],
                initializer=tf.constant_initializer(0))

            weighted = tf.matmul(inputs, weights, name='weighted')
            output = tf.add(weighted, biases, name='outputs')
            return output

    def loss(self, outputs, targets):
        with tf.name_scope('loss'):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=outputs,
                labels=targets,
                name='xentropy',
            )

            mean = tf.reduce_mean(xentropy, name='loss')
            return mean

    def train(self, loss):
        with tf.name_scope('optimize'):
            optimizer = tf.train.GradientDescentOptimizer(LR)
            return optimizer.minimize(loss)
    
    def train_data(self):
        with tf.name_scope('train_data'):
            inputs, outputs = [], []
            for m in range(0, OUTPUT_DIM*3, 3):
                inp = tf.random_normal(shape=(MBS//OUTPUT_DIM, INPUT_DIM), mean=m)
                out = tf.one_hot(indices=[m//3]*(MBS//OUTPUT_DIM), depth=OUTPUT_DIM, axis=-1)
                inputs.append(inp)
                outputs.append(out)

            inputs = tf.concat(inputs, axis=0, name='inputs')
            outputs = tf.concat(outputs, axis=0, name='outputs')
            return inputs, outputs
    
    def test_data(self):
        return self.train_data()

    def test(self, outputs, targets):
        return self.loss(outputs, targets)
