import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.structure import unpackRV

from bayesflow.conditional_dist import ConditionalDistribution
from bayesflow.elementary import Gaussian,BernoulliMatrix

def layer(inp, w, b):
    if len(inp.get_shape()) == 2:
        return tf.matmul(inp, w) + b
    else:
        return tf.pack([tf.matmul(inp_slice, w) + b for inp_slice in tf.unpack(inp)])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))

def init_zero_vector(shape):
    assert(len(shape)==1)
    n_out = shape[0]
    return tf.Variable(tf.zeros((n_out,), dtype=tf.float32))

def neural_gaussian(X, d_hidden, d_out, shape=None, **kwargs):
    augmented_shape = (2,) + shape if shape is not None else None
    encoder = NeuralGaussianTransform(X, d_hidden, d_out, shape=augmented_shape, **kwargs)
    means, stds = unpackRV(encoder)

    shape = means.shape
    return Gaussian(mean=means, std=stds, shape=shape, name="Gaussian_"+encoder.name)

def neural_bernoulli(X, d_hidden, d_out, shape=None, **kwargs):
    encoder = NeuralBernoulliTransform(X, d_hidden, d_out, shape=shape, **kwargs)
    return BernoulliMatrix(p=encoder, shape=shape)

class NeuralGaussianTransform(ConditionalDistribution):

    def __init__(self, X, d_hidden, d_z, w3=None, w4=None, w5=None, b3=None, b4=None, b5=None, **kwargs):

        self.d_hidden = d_hidden
        self.d_z = d_z        
        super(NeuralGaussianTransform, self).__init__(X=X, w3=w3, w4=w4, w5=w5, b3=b3, b4=b4, b5=b5, **kwargs)
           
    def inputs(self):
        return {"X": None, "w3": init_weights, "w4": init_weights, "w5": init_weights, "b3": init_zero_vector, "b4": init_zero_vector, "b5": init_zero_vector}

    def _input_shape(self, param, **other_shapes):
        assert (param in self.inputs().keys())
        d_x = other_shapes["X"][-1]
        if param == "w3":
            return (d_x, self.d_hidden)
        elif param in ("w4", "w5"):
            return (self.d_hidden, self.d_z)
        elif param == "b3":
            return (self.d_hidden,)
        elif param in ("b4", "b5"):
            return (self.d_z,)
        else:
            raise Exception("don't know how to produce a shape for param %s at %s" % (param, self))

    def _compute_shape(self, X_shape, w3_shape, w4_shape, w5_shape, b3_shape, b4_shape, b5_shape):
        base_shape = X_shape[:-1] + (self.d_z,)
        augmented_shape = (2,) + base_shape
        return augmented_shape
        
    def _sample(self, X, w3, w4, w5, b3, b4, b5):
        h1 = tf.nn.tanh(layer(X, w3, b3))
        mean = layer(h1, w4, b4)
        std = tf.exp(layer(h1, w5, b5))
        return tf.pack([mean, std])

    def _logp(self, result, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)

    def _entropy(self, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)

    """
    def default_q(self):
        input_qs = {inp: node.q_distribution() for (inp, node) in self.inputs_random.items()}
        input_qs.update(self.inputs_nonrandom)
        return NeuralGaussianTransform(name="q_"+self.name, d_hidden=self.d_hidden, d_z=self.d_z, **input_qs)
    """
    
class NeuralBernoulliTransform(ConditionalDistribution):
    def __init__(self, z, d_hidden, d_x, w1=None, w2=None, b1=None, b2=None, **kwargs):

        z_shape = util.extract_shape(z) if isinstance(z, tf.Tensor) else z.shape 
        self.d_z = z.shape[-1]
        self.d_hidden = d_hidden
        self.d_x = d_x

        super(NeuralBernoulliTransform, self).__init__(z=z, w1=w1, w2=w2, b1=b1, b2=b2, **kwargs)

    def inputs(self):
        return {"z": None, "w1": init_weights, "w2": init_weights, "b1": init_zero_vector, "b2": init_zero_vector}

    def _input_shape(self, param, **kwargs):
        assert (param in self.inputs().keys())
        if param == "w1":
            return (self.d_z, self.d_hidden)
        elif param == "w2":
            return (self.d_hidden, self.d_x)
        elif param == "b1":
            return (self.d_hidden,)
        elif param == "b2":
            return (self.d_x,)
        else:
            raise Exception("don't know how to produce a shape for param %s at %s" % (param, self))
    
    def _compute_shape(self, z_shape, w1_shape, w2_shape, b1_shape, b2_shape):
        return z_shape[:-1] + (self.d_x,)
    
    def _sample(self, z, w1, w2, b1, b2):
        h1 = tf.nn.tanh(layer(z, w1, b1))
        probs = tf.nn.sigmoid(layer(h1, w2, b2))
        return probs

    def _logp(self, result, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)

    def _entropy(self, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)

    def default_q(self):
        input_qs = {inp: node.q_distribution() for (inp, node) in self.inputs_random.items()}
        input_qs.update(self.inputs_nonrandom)
        return NeuralBernoulliTransform(name="q_"+self.name, d_hidden=self.d_hidden, d_x=self.d_x, **input_qs)

