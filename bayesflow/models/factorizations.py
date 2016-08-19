import numpy as np
import tensorflow as tf
import bayesflow as bf

from bayesflow.models import ConditionalDistribution
from bayesflow.parameterization import unconstrained, positive_exp

class NoisyGaussianMatrixProduct(ConditionalDistribution):
    
    def __init__(self, A, B, std, rescale=True, **kwargs):

        # optionally compute (AB' / K) instead of AB',
        # so that the marginal variance of the result equals
        # the marginal variance of the inputs
        self.rescale = rescale
        self.K = A.shape()[1]
        
        super(NoisyGaussianMatrixProduct, self).__init__(A=A, B=B, std=std,  **kwargs) 


        
    def inputs(self):
        return {"A": unconstrained, "B": unconstrained, "std": positive_exp}
        
    def _compute_shape(self, A_shape, B_shape, std_shape):
        N, K = A_shape
        M, K2 = B_shape
        assert(K == K2)
        prod_shape = (N, M)                
        return prod_shape
    
    def _compute_dtype(self, A_dtype, B_dtype, std_dtype):
        assert(A_dtype == B_dtype)
        assert(A_dtype == std_dtype)
        return A_dtype
        
    def _sample(self, A, B, std):
        shape = self.shape()
        eps = tf.placeholder(shape=shape, dtype=self.dtype)
        def random_source():
            return np.asarray(np.random.randn(*shape), dtype=np.float32)

        prod = tf.matmul(A, tf.transpose(B))
        if self.rescale:
            prod /= float(self.K)

        return {self.name: eps * std + prod}, {eps: random_source}

    def _logp(self, result, A, B, std):
        prod = tf.matmul(A, tf.transpose(B))
        lp = tf.reduce_sum(bf.dists.gaussian_log_density(result, mean=prod, stddev=std))
        return lp

    def _expected_logp(self, q_result, q_A, q_B, q_std):
        std = q_std._sampled_tf[q_std.name][0]
        var = q_result.variance + tf.square(std)
            
        mA, vA = q_A.mean, q_A.variance
        mB, vB = q_B.mean, q_B.variance

        expected_result = tf.matmul(mA, tf.transpose(mB))
        if self.rescale:
            expected_result = expected_result / self.K
        
        gaussian_lp = tf.reduce_sum(bf.dists.gaussian_log_density(q_result.mean, expected_result, variance=var))

        # can do a more efficient calculation if we assume a uniform (scalar) noise variance across all entries
        #aat_diag = tf.reduce_sum(tf.square(mA), 0)
        #p = tf.reduce_sum(vA, 0)
        #btb_diag = tf.reduce_sum(tf.square(mB), 0)
        #q = tf.reduce_sum(vB, 0)
        #correction = tf.reduce_sum(p*q + p*btb_diag + aat_diag*q) / scalar_variance

        vAvB = tf.matmul(vA, tf.transpose(vB))
        vAmB = tf.matmul(vA, tf.transpose(tf.square(mB)))
        mAvB = tf.matmul(tf.square(mA), tf.transpose(vB))
        correction = tf.reduce_sum( (vAvB + vAmB + mAvB) / var)
        if self.rescale:
            # rescaling the result is equivalent to rescaling each input by 1/sqrt(K),
            # which scales the variances (and squared means) by 1/K.
            # Which then scales each of the product terms vAvB, vAmB, etc by 1/K^2. 
            correction = correction / (self.K * self.K)
        
        expected_lp = gaussian_lp - .5 * correction
        
        return expected_lp


    """
    def elbo_term(self, symmetry_correction_hack=True):
        expected_logp, entropy = super(NoisyGaussianMatrixProduct, self).elbo_term()

        if symmetry_correction_hack:
            permutation_correction = np.sum(np.log(np.arange(1, self.K+1))) # log K!
            signflip_correction = self.K * np.log(2)
            expected_logp = expected_logp + permutation_correction + signflip_correction

        return expected_logp, entropy
    """
