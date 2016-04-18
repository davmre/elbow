import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util


from bayesflow.models import ConditionalDistribution
from bayesflow.models.q_distributions import QDistribution

def layer(inp, w, b):
    return tf.matmul(inp, w) + b

def init_weights(*shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))

def init_zero_vector(n_out):
    return tf.Variable(tf.zeros((n_out,), dtype=tf.float32))

class VAEDecoderBernoulli(ConditionalDistribution):
    def __init__(self, z, w1, w2, b1, b2, **kwargs):
        super(VAEDecoderBernoulli, self).__init__(z=z, w1=w1, w2=w2, b1=b1, b2=b2, **kwargs)

        
    def inputs(self):
        return ("z", "w1", "w2", "b1", "b2")
        
    def _compute_shape(self, z_shape, w1_shape, w2_shape, b1_shape, b2_shape):
        N, d_latent = z_shape
        d_output, = b2_shape
        return (N, d_output)
    
    def _compute_dtype(self, z_dtype, w1_dtype, w2_dtype, b1_dtype, b2_dtype):
        assert(z_dtype == w1_dtype)
        assert(z_dtype == w2_dtype)
        assert(z_dtype == b1_dtype)
        assert(z_dtype == b2_dtype)
        return z_dtype
        
    def _sample(self, A, B, std):
        raise Exception("not implemented")
    
    def _logp(self, result, z, w1, w2, b1, b2):
        h1 = tf.nn.tanh(layer(z, w1, b1))
        h2 = tf.nn.sigmoid(layer(h1, w2, b2))
        self.pred_image = h2
        obs_lp = tf.reduce_sum(bf.dists.bernoulli_log_density(result, h2))
        return obs_lp

class VAEEncoder(QDistribution):

    def __init__(self, X, d_hidden, d_z):
        N, d_x = util.extract_shape(X)
        super(VAEEncoder, self).__init__(shape=(N, d_z))

        w3 = init_weights(d_x, d_hidden)
        w4 = init_weights(d_hidden, d_z)
        w5 = init_weights(d_hidden, d_z)
        b3 = init_zero_vector(d_hidden)
        b4 = init_zero_vector(d_z)
        b5 = init_zero_vector(d_z)
        
        
        h1 = tf.nn.tanh(layer(X, w3, b3))
        self.mean = layer(h1, w4, b4)
        self.variance = tf.exp(layer(h1, w5, b5))
        self.stddev = tf.sqrt(self.variance)
        self.stochastic_eps = tf.placeholder(dtype=self.mean.dtype, shape=self.output_shape, name="eps")
        self.sample = self.stochastic_eps * self.stddev + self.mean
        self._entropy = tf.reduce_sum(bf.dists.gaussian_entropy(variance=self.variance))
        
    def sample_stochastic_inputs(self):
        sampled_eps = np.random.randn(*self.output_shape)
        return {self.stochastic_eps : sampled_eps}
    
    def entropy(self):
        return self._entropy
