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
    
    def sample_stochastic_inputs(self):
        raise Exception("not implemented")

    def entropy(self):
        raise Exception("not implemented")

    def density(self):
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
        return 0.0

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
        return 0.0

class PointwiseTransformedQDistribution(QDistribution):
    def __init__(self, parent_q, transform, implicit=False):

        
        super(PointwiseTransformedQDistribution, self).__init__(shape=parent_q.output_shape)
        self.sample, self.log_jacobian = transform(parent_q.sample)

        # an "implicit" transformation is a formal object associated with a deterministic
        # transformation in the model, where the parent q distribution is associated with
        # the untransformed variable. In this case the ELBO expectations are with respect
        # to the parent Q distribution, so the jacobian of the transformation does not appear.
        # By contrast, a non-implicit use of a transformed Q distribution would be to create
        # a new type of distribution (eg, lognormal by exponentiating a Gaussian parent)
        # that could then itself be associated with stochastic variables in the graph.
        self.implicit = implicit
        self.parent_q = parent_q
        
    def sample_stochastic_inputs(self):
        return {}
    
    def entropy(self):
        if self.implicit:
            return 0.0
        else:
            return self.parent_q.entropy() + self.log_jacobian
        
        
class GaussianQDistribution(QDistribution):
    
    def __init__(self, shape, monte_carlo_entropy=False):
        super(GaussianQDistribution, self).__init__(shape=shape)
        
        init_mean = np.float32(np.random.randn(*shape))
        init_log_stddev = np.float32(np.ones(shape) * -10)
        self.mean = tf.Variable(init_mean, name="mean")
        self.log_stddev = tf.Variable(init_log_stddev, name="log_stddev")
        self.stddev = tf.exp(self.log_stddev)
        self.variance = tf.square(self.stddev)        
        self.stochastic_eps = tf.placeholder(dtype=self.mean.dtype, shape=shape, name="eps")
        self.sample = self.stochastic_eps * self.stddev + self.mean        

        # HACK
        if monte_carlo_entropy:
            self._entropy = -(tf.reduce_sum(bf.dists.gaussian_log_density(self.sample, mean=self.mean, variance=self.variance)))
        else:
            self._entropy = tf.reduce_sum(bf.dists.gaussian_entropy(variance=self.variance))


        
    def sample_stochastic_inputs(self):
        sampled_eps = np.random.randn(*self.output_shape)
        return {self.stochastic_eps : sampled_eps}
    
    def entropy(self):
        return self._entropy
    
class BernoulliQDistribution(QDistribution):
    
    def __init__(self, shape):
        super(BernoulliQDistribution, self).__init__(shape=shape)
        
        init_log_odds = np.float32(np.random.randn(*shape))
        self.logodds = tf.Variable(init_log_odds, name="logodds")
        self.probs, _ = bf.transforms.logit(self.logodds)

        self.stochastic_eps = tf.placeholder(dtype=self.probs.dtype, shape=shape, name="eps")
        self.sample = self.stochastic_eps < self.probs     
        
        self._entropy = tf.reduce_sum(bf.dists.bernoulli_entropy(p=self.probs))
        
    def sample_stochastic_inputs(self):
        sampled_eps = np.random.rand(*self.output_shape)
        return {self.stochastic_eps : sampled_eps}
    
    def entropy(self):
        return self._entropy
