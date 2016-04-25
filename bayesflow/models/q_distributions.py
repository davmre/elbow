import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util


# TODO: do we really need separate classes to represent Q
# distributions? the "variational models" perspective is that they
# should just be (un)ConditionalDistribution objects.

class QDistribution(object):
    def __init__(self, shape):
        self.output_shape=shape

    def params(self):
        return {"sample": self.sample}
        
    def sample_stochastic_inputs(self):
        raise Exception("not implemented")

    def entropy(self):
        raise Exception("not implemented")

    def density(self):
        raise Exception("not implemented")

    def initialize_to_value(self, x):
        raise Exception("not implemented")
    
class ObservedQDistribution(QDistribution):
    def __init__(self, observed_val):
        shape = observed_val.shape
        super(ObservedQDistribution, self).__init__(shape=shape)
        
        self.mean = tf.constant(observed_val, name="mean")
        self.stddev = tf.zeros_like(self.mean, name="stddev")
        self.variance = tf.identity(self.stddev)
        self.sample = self.mean 
        
    def sample_stochastic_inputs(self):
        return {}
    
    def entropy(self):
        return tf.constant(0.0, dtype=tf.float32)

    
class DeltaQDistribution(QDistribution):
    def __init__(self, tf_value):

        shape = util.extract_shape(tf_value)
        super(DeltaQDistribution, self).__init__(shape=shape)
        
        self.mean = tf_value
        self.stddev = tf.zeros_like(self.mean, name="stddev")
        self.variance = tf.identity(self.stddev)
        self.sample = self.mean 
        
    def sample_stochastic_inputs(self):
        return {}
    
    def entropy(self):
        return tf.constant(0.0, dtype=tf.float32)
    

class GaussianQDistribution(QDistribution):
    
    def __init__(self, shape, monte_carlo_entropy=False):
        super(GaussianQDistribution, self).__init__(shape=shape)

        self.monte_carlo_entropy=monte_carlo_entropy
        
        init_mean = np.float32(np.random.randn(*shape))
        init_log_stddev = np.float32(np.ones(shape) * -10)
        self._initialize(init_mean, init_log_stddev)
        
    def _initialize(self, init_mean, init_log_stddev):
        self.mean = tf.Variable(init_mean, name="mean")
        self.log_stddev = tf.Variable(init_log_stddev, name="log_stddev")
        self.stddev = tf.exp(tf.clip_by_value(self.log_stddev, -42, 42))
        self.variance = tf.square(self.stddev)
        self.stochastic_eps = tf.placeholder(dtype=self.mean.dtype, shape=self.output_shape, name="eps")
        self.sample = self.stochastic_eps * self.stddev + self.mean        

        # HACK
        if self.monte_carlo_entropy:
            self._entropy = -(tf.reduce_sum(bf.dists.gaussian_log_density(self.sample, mean=self.mean, variance=self.variance)))
        else:
            self._entropy = tf.reduce_sum(bf.dists.gaussian_entropy(variance=self.variance))
        
    def params(self):
        return {"mean": self.mean, "stddev": self.stddev, "sample": self.sample}
        
    def sample_stochastic_inputs(self):
        sampled_eps = np.random.randn(*self.output_shape)
        return {self.stochastic_eps : sampled_eps}
    
    def entropy(self):
        return self._entropy

    def initialize_to_value(self, x):
        init_log_stddev = np.float32(np.ones(self.output_shape) * -4)
        self._initialize(x, init_log_stddev)

    
class BernoulliQDistribution(QDistribution):
    
    def __init__(self, shape):
        super(BernoulliQDistribution, self).__init__(shape=shape)
        
        init_log_odds = np.float32(np.random.randn(*shape))
        self._initialize(init_log_odds)
        
    def _initialize(self, logodds):
        self.logodds = tf.Variable(logodds, name="logodds")
        self.probs, _ = bf.transforms.logit(self.logodds)

        self.stochastic_eps = tf.placeholder(dtype=self.probs.dtype, shape=self.output_shape, name="eps")
        self.sample = self.stochastic_eps < self.probs     
        
        self._entropy = tf.reduce_sum(bf.dists.bernoulli_entropy(p=self.probs))
        
    def params(self):
        return {"probs": self.probs, "sample": self.sample}
        
    def sample_stochastic_inputs(self):
        sampled_eps = np.random.rand(*self.output_shape)
        return {self.stochastic_eps : sampled_eps}
    
    def entropy(self):
        return self._entropy

    def initialize_to_value(self, x):
        # assign logodds of 5.0 to True and -5.0 to False
        implied_logodds = 5.0 * (x*2.0-1.0)
        self._initialize(implied_logodds)
        #self.logodds.assign(implied_logodds)

    
class SimplexQDistribution(QDistribution):
    # define a distribution on the simplex as a transform of an
    # (n-1)-dimensional Gaussian
    
    def __init__(self, n, monte_carlo_entropy=True):
        super(SimplexQDistribution, self).__init__(shape=(n,))
        self.n = n
        
        if n > 1:
            self.gaussian_q = GaussianQDistribution(shape=(n-1,), monte_carlo_entropy=monte_carlo_entropy)
            self._simplex_input = tf.concat(0, [self.gaussian_q.sample, tf.zeros((1,), dtype=tf.float32)])
            self.sample, self.log_jacobian = bf.transforms.simplex(self._simplex_input)
                    
            self._entropy = self.gaussian_q.entropy() + self.log_jacobian
        else:
            self.gaussian_q = None
            self.sample = tf.ones((1,))
            self._entropy = tf.constant(0.0)
                
    def sample_stochastic_inputs(self):
        if self.gaussian_q is not None:
            return self.gaussian_q.sample_stochastic_inputs()
        else:
            return {}
        
    def entropy(self):
        return self._entropy

    def initialize_to_value(self, x):
        if self.gaussian_q is not None:
            implied_log = np.log(x)
            shifted = implied_log - implied_log[-1]
            self.gaussian_q.initialize_to_value(shifted[:-1])
        
