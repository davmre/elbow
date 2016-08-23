import numpy as np
import tensorflow as tf
import bayesflow as bf
import bayesflow.util as util

from bayesflow.conditional_dist import ConditionalDistribution
from bayesflow.parameterization import unconstrained, positive_exp, simplex_constrained

class NoisyGaussianMatrixProduct(ConditionalDistribution):
    
    def __init__(self, A, B, std, rescale=True, **kwargs):

        # optionally compute (AB' / K) instead of AB',
        # so that the marginal variance of the result equals
        # the marginal variance of the inputs
        self.rescale = rescale
        self.K = A.shape[1]
        
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
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)

        prod = tf.matmul(A, tf.transpose(B))
        if self.rescale:
            prod /= float(self.K)

        return eps * std + prod

    def _logp(self, result, A, B, std):
        prod = tf.matmul(A, tf.transpose(B))
        lp = tf.reduce_sum(util.dists.gaussian_log_density(result, mean=prod, stddev=std))
        return lp

    def _expected_logp(self, q_result, q_A, q_B, q_std=None):

        std = q_std._sampled if q_std is not None else self.inputs_nonrandom['std']
        
        var = q_result.variance + tf.square(std)
            
        mA, vA = q_A.mean, q_A.variance
        mB, vB = q_B.mean, q_B.variance

        expected_result = tf.matmul(mA, tf.transpose(mB))
        if self.rescale:
            expected_result = expected_result / self.K
        
        gaussian_lp = tf.reduce_sum(util.dists.gaussian_log_density(q_result.mean, expected_result, variance=var))

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

class NoisyCumulativeSum(ConditionalDistribution):
    
    def __init__(self, A, std,  **kwargs):
        super(NoisyCumulativeSum, self).__init__(A=A, std=std,  **kwargs)      
        
    def inputs(self):
        return {"A": unconstrained,  "std": positive_exp}

    def _compute_shape(self, A_shape, std_shape): 
        return A_shape
    
    def _compute_dtype(self, A_dtype, std_dtype):
        assert(A_dtype == std_dtype)
        return A_dtype
    
    def _sample(self, A, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        return eps * std + tf.cumsum(A)

    def _expected_logp(self, q_result, q_A, q_std=None):
        
        std = q_std._sampled if q_std is not None else self.inputs_nonrandom['std']
        
        var = q_result.variance + tf.square(std)
        X = q_result.mean
        
        # TODO: hugely inefficent hack 
        #N, D = self.output_shape
        #cumsum_mat = np.float32(np.tril(np.ones((N, N))))
        #r = np.float32(np.arange(N, 0, -1)).reshape((1, -1))

        expected_X = tf.cumsum(q_A.mean)
        gaussian_lp = util.dists.gaussian_log_density(X, expected_X, variance=var)

        # performs a reverse cumulative sum
        #R = tf.matmul(cumsum_mat, 1.0/var, transpose_a=True)
        rvar = tf.reverse(1.0/var, [True, False])
        R = tf.reverse(tf.cumsum(rvar), [True, False])
        corrections = -.5 * R * q_A.variance

        reduced_gaussian_lp =  tf.reduce_sum(gaussian_lp) 
        reduced_correction = tf.reduce_sum(corrections)
        return reduced_gaussian_lp + reduced_correction
    
    def _logp(self, result, A, std):
        N, D = self.shape
        cumsum_mat = np.float32(np.tril(np.ones((N, N))))

        expected_X = tf.cumsum(A)
        var = tf.square(std)
        gaussian_lp = util.dists.gaussian_log_density(result, expected_X, variance=var)
        return tf.reduce_sum(gaussian_lp)

class GMMClustering(ConditionalDistribution):

    def __init__(self, weights, centers, std, **kwargs):
        self.n_clusters = centers.shape[0]
        super(GMMClustering, self).__init__(weights=weights, centers=centers, std=std, **kwargs)
        
    def inputs(self):
        return {"weights": simplex_constrained, "centers": unconstrained, "std": positive_exp}

    def _compute_shape(self, weights_shape, centers_shape, std_shape):
        raise Exception("cannot infer shape for GMMClustering, must specify number of cluster draws")
    
    def _compute_dtype(self, weights_dtype, centers_dtype, std_dtype):
        assert(centers_dtype == std_dtype)
        return centers_dtype
    
    def _sample(self, weights, centers, std):
        N, D = self.shape
        K = self.n_clusters

        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        center_idxs = tf.squeeze(tf.multinomial(tf.log(tf.expand_dims(weights, 0)), num_samples=N), squeeze_dims=(0,))
        
        chosen_centers = tf.gather(centers, center_idxs)
        return chosen_centers + eps*std

    def _logp(self, result, weights, centers, std):
        total_logps = None
        
        # loop over clusters
        for i, center in enumerate(tf.unpack(centers)):
            # compute vector of likelihoods that each point could be generated from *this* cluster
            cluster_lls = tf.reduce_sum(util.dists.gaussian_log_density(result, center, std), 1)

            # sum these likelihoods, weighted by cluster probabilities
            cluster_logps = tf.log(weights[i]) + cluster_lls
            if total_logps is not None:
                total_logps = util.logsumexp(total_logps, cluster_logps)
            else:
                total_logps = cluster_logps
            
        # finally sum the log probabilities of all points to get a likelihood for the dataset
        obs_lp = tf.reduce_sum(total_logps)
        
        return obs_lp

class NoisyLatentFeatures(ConditionalDistribution):

    def __init__(self, B, G, std,  **kwargs):
        super(NoisyLatentFeatures, self).__init__(B=B, G=G, std=std,  **kwargs)
        self.K = B.shape[1]
        
    def inputs(self):
        return {"B": None, "G": unconstrained, "std": positive_exp}

    def _compute_shape(self, B_shape, G_shape, std_shape):
        N, K = B_shape
        K2, M = G_shape
        assert(K == K2)
        prod_shape = (N, M)                
        return prod_shape
    
    def _compute_dtype(self, B_dtype, G_dtype, std_dtype):
        assert(G_dtype == std_dtype)
        return G_dtype
    
    def _sample(self, B, G, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)        
        return tf.matmul(B, G) + eps*std
    
    def _expected_logp(self, q_result, q_B, q_G, q_std=None):
        """ 
        compute E_Q{X, G, B} [ log p(X | G, B) ]
          where q(G) ~ N(q_G_means, q_G_stds)
                q(B) ~ Bernoulli(bernoulli_params)
                q(X) ~ N(q_X_means, q_X_stds)
                       (a posterior on X representing an upwards message.
                       In the object-level case the variances are zero)
          and the model itself is
              log p(X | G, B) ~ N(X; BG, noise_stds)
          i.e. each entry of X is modeled as a Gaussian with mean given 
          by the corresponding entry of ZG, and stddev given by the 
          corresponding entry of noise_stds. (in principle these can all
          be different though in practice we will usually have a single
          global stddev, or perhaps per-column or per-row stddevs). 


        Matrix shapes: N datapoints, D dimensions, K latent features
         X: N x D
         G: K x D
         B: N x K
        """

        std = q_std._sampled if q_std is not None else self.inputs_nonrandom["std"]
        var = q_result.variance + tf.square(std)
        X_means = q_result.mean
        bernoulli_params = q_B.probs
        
        expected_X = tf.matmul(bernoulli_params, q_G.mean)
        precisions = 1.0/var
        gaussian_lp = util.dists.gaussian_log_density(X_means, expected_X, variance=var)

        mu2 = tf.square(q_G.mean)
        tau_V = tf.matmul(bernoulli_params, q_G.variance)
        tau_tau2_mu2 = tf.matmul(bernoulli_params - tf.square(bernoulli_params), mu2)
        tmp = tau_V + tau_tau2_mu2
        lp_correction = tmp * precisions

        pointwise_expected_lp = gaussian_lp - .5*lp_correction 
        expected_lp = tf.reduce_sum(pointwise_expected_lp)

        return expected_lp

    def _entropy(self, **kwargs):
        # not yet implemented
        return tf.constant(np.nan)

    """
    def elbo_term(self, symmetry_correction_hack=True):
        expected_logp, entropy = super(NoisyLatentFeatures, self).elbo_term()

        if symmetry_correction_hack:
            permutation_correction = np.sum(np.log(np.arange(1, self.K+1))) # log K!
            expected_logp = expected_logp + permutation_correction
                
        return expected_logp, entropy
    """

class MultiplicativeGaussianNoise(ConditionalDistribution):    

    def __init__(self, A, std, **kwargs):
        super(MultiplicativeGaussianNoise, self).__init__(A=A, std=std, **kwargs)   

    def inputs(self):
        return {"A": unconstrained, "std": positive_exp}

    def _compute_shape(self, A_shape, std_shape):
        return A_shape
    
    def _compute_dtype(self, A_dtype, std_dtype):
        assert(A_dtype == std_dtype)
        return A_dtype
    
    def _sample(self, A, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        noise = eps * std
        return A * noise

    def _logp(self, result, A, std):
        residuals = result / A
        lp = tf.reduce_sum(util.dists.gaussian_log_density(residuals, 0.0, std))
        return lp

