import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution

import scipy.stats

class GammaMatrix(ConditionalDistribution):
    
    def __init__(self, alpha, beta, **kwargs):
        super(GammaMatrix, self).__init__(alpha=alpha, beta=beta, **kwargs)
    
    def inputs(self):
        return ("alpha", "beta")
    
    def _sample(self, alpha, beta):
        scale = 1.0/beta
        return np.asarray(scipy.stats.gamma(a=alpha, scale=scale).rvs(*self.output_shape), dtype=self.dtype)

    def _logp(self, result, alpha, beta):    
        lp = tf.reduce_sum(bf.dists.gamma_log_density(result, alpha, beta))
        return lp

    def _compute_shape(self, alpha_shape, beta_shape):
        return bf.util.broadcast_shape(alpha_shape, beta_shape)
        
    def _compute_dtype(self, alpha_dtype, beta_dtype):
        assert(alpha_dtype==beta_dtype)
        return alpha_dtype

class BernoulliMatrix(ConditionalDistribution):
    def __init__(self, p, **kwargs):
        super(BernoulliMatrix, self).__init__(p=p, **kwargs)        
        
    def inputs(self):
        return ("p")
        
    def _sample(self, p):
        N, K = self.output_shape
        B = np.asarray(np.random.rand(N, K) < p, dtype=self.dtype)
        return B
    
    def _expected_logp(self, q_result, q_p):
        # compute E_q [log p(z)] for a given prior p(z) and an approximate posterior q(z).
        # note q_p represents our posterior uncertainty over the parameters p: if these are known and
        # fixed, q_p is just a delta function, otherwise we have to do Monte Carlo sampling.
        # Whereas q_z represents our posterior over the sampled (Bernoulli) values themselves,
        # and we assume this is in the form of a set of Bernoulli probabilities. 

        p_z = q_p.sample
        q_z = q_result.probs
        
        lp = -tf.reduce_sum(bf.dists.bernoulli_entropy(q_z, cross_q = p_z))
        return lp
    
    def _compute_shape(self, p_shape):
        return p_shape
        
    def _compute_dtype(self, p_dtype):
        return np.int32
    
    
class MultinomialMatrix(ConditionalDistribution):
    # matrix in which each row contains a single 1, and 0s otherwise.
    # the location of the 1 is sampled from a multinomial(p) distribution,
    # where the parameter p is a (normalized) vector of probabilities.
    # matrix shape: N x K
    # parameter shape: (K,)
    
    def __init__(self, p, **kwargs):
        super(MultinomialMatrix, self).__init__(p=p, **kwargs) 
        
    def inputs(self):
        return ("p")
        
    def _sample(self, p):
        N, K = self.output_shape
        choices = np.random.choice(np.arange(K), size=(N,),  p=p)
        M = np.zeros(N, K, dtype=self.dtype)
        r = np.arange(N)
        M[r, choices] = 1
        return M
    
    def _expected_logp(self, q_result, q_p):
        p = q_p.sample
        q = q_result.probs

        lp = tf.reduce_sum(bf.dists.multinomial_entropy(q, cross_q=p))
        return lp
    
    def _compute_shape(self, p_shape):
        return p_shape
        
    def _compute_dtype(self, p_dtype):
        return np.int32
    
class GaussianMatrix(ConditionalDistribution):
    
    def __init__(self, mean, std, **kwargs):
        super(GaussianMatrix, self).__init__(mean=mean, std=std, **kwargs) 
        
    def inputs(self):
        return ("mean", "std")
        
    def _sample(self, mean, std):
        return np.asarray(np.random.randn(*self.output_shape), dtype=self.dtype) * std + mean
    
    def _logp(self, result, mean, std):
        lp = tf.reduce_sum(bf.dists.gaussian_log_density(result, mean = mean, stddev=std))
        return lp
    
    def _expected_logp(self, q_result, q_mean, q_std):
        cross = bf.dists.gaussian_cross_entropy(q_result.mean, q_result.variance, q_mean.sample, tf.square(q_std.sample))
        return -tf.reduce_sum(cross)
    
    def _compute_shape(self, mean_shape, std_shape):
        return bf.util.broadcast_shape(mean_shape, std_shape)
        
    def _compute_dtype(self, mean_dtype, std_dtype):
        assert(mean_dtype==std_dtype)
        return mean_dtype
        
