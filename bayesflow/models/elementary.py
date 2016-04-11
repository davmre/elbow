import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution

class GammaMatrix(ConditionalDistribution):
    
    def __init__(self, alpha, beta, output_shape=None):
        super(GaussianMatrix, self).__init__(alpha=alpha, beta=beta, output_shape=output_shape)
    
    def inputs(self):
        return ("alpha", "beta")
    
    def _sample(self, alpha, beta):
        scale = 1.0/beta
        return np.asarray(scipy.stats.gamma(alpha=alpha, scale=scale).rvs(*self.output_shape), dtype=self.dtype)
        
    def _expected_logp(self, q_result, q_alpha, q_beta):
        alpha = q_alpha.sample
        beta = q_beta.sample
        result = q_result.sample
        
        lp = tf.reduce_sum(bf.dists.gamma_log_density(result, alpha, beta))
        return lp

    def _compute_shape(self, alpha_shape, beta_shape):
        return bf.util.broadcast_shape(alpha_shape, beta_shape)
        
    def _compute_dtype(self, alpha_dtype, beta_dtype):
        assert(alpha_dtype==beta_dtype)
        return alpha_dtype
    
class GaussianMatrix(ConditionalDistribution):
    
    def __init__(self, mean, std, output_shape=None):
        super(GaussianMatrix, self).__init__(mean=mean, std=std, output_shape=output_shape)        
        
    def inputs(self):
        return ("mean", "std")
        
    def _sample(self, mean, std):
        return np.asarray(np.random.randn(*self.output_shape), dtype=self.dtype) * std + mean
    
    def _expected_logp(self, q_result, q_mean, q_std):
        # E log N(result ; 0, column_precs )
        std = q_std.sample
        var = q_result.variance + q_mean.variance + tf.square(std)
        lp = tf.reduce_sum(bf.dists.gaussian_log_density(q_result.mean, mean = q_mean.mean, variance=var))
        return lp
    
    def _compute_shape(self, mean_shape, std_shape):
        return bf.util.broadcast_shape(mean_shape, std_shape)
        
    def _compute_dtype(self, mean_dtype, std_dtype):
        assert(mean_dtype==std_dtype)
        return mean_dtype
        
