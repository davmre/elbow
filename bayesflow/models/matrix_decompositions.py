import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution

class NoisyGaussianMatrixProduct(ConditionalDistribution):
    
    def __init__(self, A, B, std):
        super(NoisyGaussianMatrixProduct, self).__init__(A=A, B=B, std=std)      
        
    def inputs(self):
        return ("A", "B", "std")
        
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
        noise = np.asarray(np.random.randn(*self.output_shape), dtype=self.dtype) * std
        return np.dot(A, B.T) + noise
    
    def _expected_logp(self, q_result, q_A, q_B, q_std):
        std = q_std.sample
        var = q_result.variance + tf.square(std)
            
        mA, vA = q_A.mean, q_A.variance
        mB, vB = q_B.mean, q_B.variance
        
        expected_result = tf.matmul(mA, tf.transpose(mB))
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
        
        expected_lp = gaussian_lp - .5 * correction
        
        return expected_lp
    
class NoisyCumulativeSum(ConditionalDistribution):
    
    def __init__(self, A, std):
        super(NoisyCumulativeSum, self).__init__(A=A, std=std)      
        
    def inputs(self):
        return ("A",  "std")

    def _compute_shape(self, A_shape, std_shape): 
        return A_shape
    
    def _compute_dtype(self, A_dtype, std_dtype):
        assert(A_dtype == std_dtype)
        return A_dtype
    
    def _sample(self, A, std):
        noise = np.asarray(np.random.randn(*self.output_shape), self.dtype) * std
        return np.cumsum(A, axis=0) + noise
    
    def _expected_logp(self, q_result, q_A, q_std):
        
        
        std = q_std.sample
        var = q_result.variance + tf.square(std)
        X = q_result.mean
        
        # TODO: hugely inefficent hack 
        N, D = self.output_shape
        cumsum_mat = np.float32(np.tril(np.ones((N, N))))
        r = np.float32(np.arange(N, 0, -1)).reshape((1, -1))

        expected_X = tf.matmul(cumsum_mat, q_A.mean)
        gaussian_lp = bf.dists.gaussian_log_density(X, expected_X, variance=var)

        # performs a reverse cumulative sum
        R = tf.matmul(cumsum_mat, 1.0/var, transpose_a=True)
        corrections = -.5 * R * q_A.variance

        reduced_gaussian_lp =  tf.reduce_sum(gaussian_lp) 
        reduced_correction = tf.reduce_sum(corrections)
        return reduced_gaussian_lp + reduced_correction
    
