import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util


from bayesflow.models import ConditionalDistribution
from bayesflow.models.q_distributions import QDistribution

def layer(inp, w, b):
    if len(inp.get_shape()) == 2:
        return tf.matmul(inp, w) + b
    else:
        return tf.pack([tf.matmul(inp_slice, w) + b for inp_slice in tf.unpack(inp)])

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
        d_latent = z_shape[-1]
        d_output, = b2_shape
        return z_shape[:-1] + (d_output,)
    
    def _compute_dtype(self, z_dtype, w1_dtype, w2_dtype, b1_dtype, b2_dtype):
        assert(z_dtype == w1_dtype)
        assert(z_dtype == w2_dtype)
        assert(z_dtype == b1_dtype)
        assert(z_dtype == b2_dtype)
        return z_dtype
        
    def _sample(self, z, w1, w2, b1, b2, return_probs=False):

        g = tf.Graph()
        with g.as_default():
            z = tf.convert_to_tensor(z)
            w1 = tf.convert_to_tensor(w1)
            w2 = tf.convert_to_tensor(w2)
            b1 = tf.convert_to_tensor(b1)
            b2 = tf.convert_to_tensor(b2)

            h1 = tf.nn.tanh(layer(z, w1, b1))
            h2 = tf.nn.sigmoid(layer(h1, w2, b2))

            init = tf.initialize_all_variables()
            
        sess = tf.Session(graph=g)
        sess.run(init)
        probs = sess.run(h2)

        if return_probs:
            return probs
        else:
            return np.asarray(np.random.rand(*probs.shape) < probs, dtype=self.dtype)
    
    def _logp(self, result, z, w1, w2, b1, b2):
        h1 = tf.nn.tanh(layer(z, w1, b1))
        h2 = tf.nn.sigmoid(layer(h1, w2, b2))
        self.pred_image = h2
        obs_lp = tf.reduce_sum(bf.dists.bernoulli_log_density(result, h2))
        return obs_lp

class VAEEncoder(QDistribution):

    def __init__(self, X, d_hidden, d_z):
        x_shape = util.extract_shape(X)
        d_x = x_shape[-1]
        z_shape = x_shape[:-1] + (d_z,)
        super(VAEEncoder, self).__init__(shape=z_shape)

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
