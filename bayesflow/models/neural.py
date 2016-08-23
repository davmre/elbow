import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.conditional_dist import ConditionalDistribution

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

class NeuralGaussian(ConditionalDistribution):

    def __init__(self, X, d_hidden, d_z, w3=None, w4=None, w5=None, b3=None, b4=None, b5=None, **kwargs):

        x_shape = util.extract_shape(X) if isinstance(X, tf.Tensor) else X.shape 
        self.d_x = x_shape[-1]
        self.d_hidden = d_hidden
        self.d_z = d_z
        
        super(NeuralGaussian, self).__init__(X=X, w3=w3, w4=w4, w5=w5, b3=b3, b4=b4, b5=b5, **kwargs)
           
    def inputs(self):
        return {"X": None, "w3": init_weights, "w4": init_weights, "w5": init_weights, "b3": init_zero_vector, "b4": init_zero_vector, "b5": init_zero_vector}

    def _input_shape(self, param):
        assert (param in self.inputs().keys())
        if param == "w3":
            return (self.d_x, self.d_hidden)
        elif param in ("w4", "w5"):
            return (self.d_hidden, self.d_z)
        elif param == "b3":
            return (self.d_hidden,)
        elif param in ("b4", "b5"):
            return (self.d_z,)
        else:
            raise Exception("don't know how to produce a shape for param %s at %s" % (param, self))

    def _compute_shape(self, X_shape, w3_shape, w4_shape, w5_shape, b3_shape, b4_shape, b5_shape):
        return X_shape[:-1] + (self.d_z,)
        
    def _build_network(self, X, w3, w4, w5, b3, b4, b5):
        h1 = tf.nn.tanh(layer(X, w3, b3))
        mean = layer(h1, w4, b4)
        variance = tf.exp(layer(h1, w5, b5))
        return mean, variance

    def _setup_canonical_sample(self):
        input_samples = {}
        for param, node in self.inputs_random.items():
            input_samples[param] = node._sampled
        for param, tensor in self.inputs_nonrandom.items():
            input_samples[param] = tensor

        # more efficient to re-use the same network for sampled value and entropy
        mean, variance = self._build_network(**input_samples)
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)        
        self._sampled = mean + eps * tf.sqrt(variance)
        self._sampled_entropy = tf.reduce_sum(util.dists.gaussian_entropy(variance=variance))
        
    def _sample(self, **kwargs):
        mean, variance = self._build_network(**kwargs)
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        return mean + eps * tf.sqrt(variance)
    
    def _entropy(self, **kwargs):
        mean, variance = self._build_network(**kwargs)
        return tf.reduce_sum(util.dists.gaussian_entropy(variance=variance))

class NeuralBernoulli(ConditionalDistribution):
    def __init__(self, z, d_hidden, d_x, w1=None, w2=None, b1=None, b2=None, **kwargs):

        z_shape = util.extract_shape(z) if isinstance(z, tf.Tensor) else z.shape 
        self.d_z = z.shape[-1]
        self.d_hidden = d_hidden
        self.d_x = d_x

        super(NeuralBernoulli, self).__init__(z=z, w1=w1, w2=w2, b1=b1, b2=b2, **kwargs)

    def inputs(self):
        return {"z": None, "w1": init_weights, "w2": init_weights, "b1": init_zero_vector, "b2": init_zero_vector}

    def _input_shape(self, param):
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
    
    def _build_network(self, z, w1, w2, b1, b2):
        h1 = tf.nn.tanh(layer(z, w1, b1))
        probs = tf.nn.sigmoid(layer(h1, w2, b2))
        return probs

    def _setup_canonical_sample(self):
        input_samples = {}
        for param, node in self.inputs_random.items():
            input_samples[param] = node._sampled
        for param, tensor in self.inputs_nonrandom.items():
            input_samples[param] = tensor
        
        # more efficient to re-use the same network for sampled value and entropy
        probs = self._build_network(**input_samples)
        unif = tf.random_uniform(shape=self.shape)        
        self._sampled = unif < probs
        self._sampled_entropy = tf.reduce_sum(util.dists.bernoulli_entropy(p=probs))
    
    def _sample(self, return_probs=False, **kwargs):
        probs = self._build_network(**kwargs)
        if return_probs:
            return probs
        else:
            unif = tf.random_uniform(shape=probs.shape)
            return unif < probs
    
    def _logp(self, result, **kwargs):
        probs = self._build_network(**kwargs)
        obs_lp = tf.reduce_sum(util.dists.bernoulli_log_density(result, probs))
        return obs_lp

    def _entropy(self, **kwargs):
        probs = self._build_network(**kwargs)
        return tf.reduce_sum(util.dists.bernoulli_entropy(p=probs))
