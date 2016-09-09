import numpy as np
import tensorflow as tf
import bayesflow as bf
import bayesflow.util as util

from bayesflow.elementary import Gaussian
from bayesflow.conditional_dist import ConditionalDistribution
from bayesflow.parameterization import unconstrained, positive_exp, simplex_constrained

class NoisyGaussianMatrixProduct(ConditionalDistribution):
    
    def __init__(self, A, B, std=None, rescale=True, **kwargs):

        # optionally compute (AB' / K) instead of AB',
        # so that the marginal variance of the result equals
        # the marginal variance of the inputs
        self.rescale = rescale
        self.K = A.shape[1]
        
        super(NoisyGaussianMatrixProduct, self).__init__(A=A, B=B, std=std,  **kwargs) 
        
    def inputs(self):
        return {"A": unconstrained, "B": unconstrained, "std": positive_exp}

    def derived_parameters(self, A, B, std, **kwargs):
        derived = {}
        derived["mean"] = tf.matmul(A, tf.transpose(B))
        if self.rescale:
            derived["mean"] /= self.K
        derived["variance"] = std**2
        return derived
    
    def _compute_shape(self, A_shape, B_shape, std_shape):
        N, K = A_shape
        M, K2 = B_shape
        assert(K == K2)
        prod_shape = (N, M)                
        return prod_shape
    
    def _sample(self, A, B, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)

        prod = tf.matmul(A, tf.transpose(B))
        if self.rescale:
            prod /= float(self.K)

        return eps * std + prod

    def _logp(self, result, A, B, std):
        prod = tf.matmul(A, tf.transpose(B))
        if self.rescale:
            prod = prod / self.K
        lp = tf.reduce_sum(util.dists.gaussian_log_density(result, mean=prod, stddev=std))
        return lp

    def _expected_logp(self, q_result, q_A=None, q_B=None, q_std=None):

        std = q_std._sampled if q_std is not None else self.inputs_nonrandom['std']
        
        try:
            var = q_result.variance + tf.square(std)
            mA, vA = q_A.mean, q_A.variance
            mB, vB = q_B.mean, q_B.variance
        except:
            # if any Q dists are missing or nongaussian
            A = q_A._sampled if q_A is not None else self.inputs_nonrandom['A']
            B = q_B._sampled if q_B is not None else self.inputs_nonrandom['B']
            return self._logp(result=q_result._sampled, A=A, B=B, std=std)
            
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

    def default_q(self):
        if "A" in self.inputs_random:
            q_A = self.inputs_random["A"].q_distribution()
        else:
            q_A = self.inputs_nonrandom["A"]

        if "B" in self.inputs_random:
            q_B = self.inputs_random["B"].q_distribution()
        else:
            q_B = self.inputs_nonrandom["B"]

        std = positive_exp(shape=self.shape)
        return NoisyGaussianMatrixProduct(A=q_A, B=q_B, std=std,
                                          rescale=self.rescale,
                                          shape=self.shape, name="q_"+self.name)

    def _hack_symmetry_correction(self):
        # TODO replace this with the correct area for the Stiefel manifold
        permutation_correction = np.sum(np.log(np.arange(1, self.K+1))) # log K!
        signflip_correction = self.K * np.log(2)
        return permutation_correction + signflip_correction


class NoisySparseGaussianMatrixProduct(ConditionalDistribution):
    
    def __init__(self, A, B, std=None, row_idxs=None, col_idxs=None, rescale=True, **kwargs):

        # optionally compute (AB' / K) instead of AB',
        # so that the marginal variance of the result equals
        # the marginal variance of the inputs
        self.rescale = rescale
        self.K = A.shape[1]
        
        super(NoisySparseGaussianMatrixProduct, self).__init__(A=A, B=B, std=std, row_idxs=row_idxs, col_idxs=col_idxs, **kwargs) 
        
    def inputs(self):
        return {"A": unconstrained, "B": unconstrained, "std": positive_exp, "row_idxs": None, "col_idxs": None}

    def derived_parameters(self, A, B, std, row_idxs, col_idxs, **kwargs):
        derived = {}
        Aidx = tf.gather(A, row_idxs)
        Bidx = tf.gather(B, col_idxs)
        prod = tf.reduce_sum(Aidx * Bidx, 1)

        if self.rescale:
            prod = prod / self.K
        
        derived["mean"] = prod
        derived["variance"] = std**2
        return derived
    
    def _compute_shape(self, A_shape, B_shape, std_shape, row_idxs_shape, col_idxs_shape):
        N, K = A_shape
        M, K2 = B_shape
        assert(K == K2)

        (nidxs,) = row_idxs_shape
        (nidxs2,) = col_idxs_shape
        assert( nidxs == nidxs2)
        return (nidxs,)
    
    def _sample(self, A, B, std, row_idxs, col_idxs):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)

        Aidx = tf.gather(A, row_idxs)
        Bidx = tf.gather(B, col_idxs)
        prod = tf.reduce_sum(Aidx * Bidx, 1)
        
        if self.rescale:
            prod /= float(self.K)

        return eps * std + prod

    def _logp(self, result, A, B, std, row_idxs, col_idxs):

        Aidx = tf.gather(A, row_idxs)
        Bidx = tf.gather(B, col_idxs)
        prod = tf.reduce_sum(Aidx * Bidx, 1)
        if self.rescale:
            prod = prod / self.K
        lp = tf.reduce_sum(util.dists.gaussian_log_density(result, mean=prod, stddev=std))
        return lp

    def _expected_logp(self, q_result, q_A=None, q_B=None, q_std=None, q_row_idxs=None, q_col_idxs=None):

        std = q_std._sampled if q_std is not None else self.inputs_nonrandom['std']
        row_idxs = q_row_idxs._sampled if q_row_idxs is not None else self.inputs_nonrandom['row_idxs']
        col_idxs = q_col_idxs._sampled if q_col_idxs is not None else self.inputs_nonrandom['col_idxs']

        try:
            var = q_result.variance + tf.square(std)
            mA, vA = tf.gather(q_A.mean, row_idxs), tf.gather(q_A.variance, row_idxs)
            mB, vB = tf.gather(q_B.mean, col_idxs), tf.gather(q_B.variance, col_idxs)
        except Exception as e:
            # if any Q dists are missing or nongaussian
            print "devolving to stochastic logp", e
            A = q_A._sampled if q_A is not None else self.inputs_nonrandom['A']
            B = q_B._sampled if q_B is not None else self.inputs_nonrandom['B']
            return self._logp(result=q_result._sampled, A=A, B=B, std=std, row_idxs=row_idxs, col_idxs=col_idxs)
            
        expected_result = tf.reduce_sum(mA * mB, 1)
        if self.rescale:
            expected_result = expected_result / self.K
        
        gaussian_lp = tf.reduce_sum(util.dists.gaussian_log_density(q_result.mean, expected_result, variance=var))

        vAvB = tf.reduce_sum(vA * vB, 1)
        vAmB = tf.reduce_sum(vA * tf.square(mB), 1)
        mAvB = tf.reduce_sum(tf.square(mA) * vB, 1)
        correction = tf.reduce_sum( (vAvB + vAmB + mAvB) / var)
        if self.rescale:
            # rescaling the result is equivalent to rescaling each input by 1/sqrt(K),
            # which scales the variances (and squared means) by 1/K.
            # Which then scales each of the product terms vAvB, vAmB, etc by 1/K^2. 
            correction = correction / (self.K * self.K)
        
        expected_lp = gaussian_lp - .5 * correction
        
        return expected_lp

    def _hack_symmetry_correction(self):
        # TODO replace this with the correct area for the Stiefel manifold
        permutation_correction = np.sum(np.log(np.arange(1, self.K+1))) # log K!
        signflip_correction = self.K * np.log(2)
        return permutation_correction + signflip_correction
    
class NoisyCumulativeSum(ConditionalDistribution):
    
    def __init__(self, A, std,  **kwargs):
        super(NoisyCumulativeSum, self).__init__(A=A, std=std,  **kwargs)      
        
    def inputs(self):
        return {"A": unconstrained,  "std": positive_exp}

    def _compute_shape(self, A_shape, std_shape): 
        return A_shape
    
    def _sample(self, A, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        return eps * std + tf.cumsum(A)

    def _expected_logp(self, q_result, q_A, q_std=None):
        
        std = q_std._sampled if q_std is not None else self.inputs_nonrandom['std']

        try:
            A_mean = q_A.mean
            A_variance = q_A.variance

            var = q_result.variance + tf.square(std)
            X = q_result.mean
        except:
            A = q_A._sampled if q_A is not None else self.inputs_nonrandom['A']
            return self._logp(result=q_result._sampled, A=A, std=std)
            
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

    def derived_parameters(self, A, std, **kwargs):
        derived = {}
        derived["variance"] = std**2
        derived["mean"] = tf.cumsum(A)
        return derived

    def default_q(self):
        if "A" in self.inputs_random:
            q_A = self.inputs_random["A"].q_distribution()
        else:
            q_A = self.inputs_nonrandom["A"]

        std = positive_exp(shape=self.shape)
        return NoisyCumulativeSum(A=q_A, std=std, shape=self.shape, name="q_"+self.name)
    
class GMMClustering(ConditionalDistribution):

    def __init__(self, weights, centers, std, **kwargs):
        self.n_clusters = centers.shape[0]
        super(GMMClustering, self).__init__(weights=weights, centers=centers, std=std, **kwargs)
        
    def inputs(self):
        return {"weights": simplex_constrained, "centers": unconstrained, "std": positive_exp}

    def _compute_shape(self, weights_shape, centers_shape, std_shape):
        raise Exception("cannot infer shape for GMMClustering, must specify number of cluster draws")
    
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

    def default_q(self):
        if "weights" in self.inputs_random:
            q_weights = self.inputs_random["weights"].q_distribution()
        else:
            q_weights = self.inputs_nonrandom["weights"]

        if "centers" in self.inputs_random:
            q_centers = self.inputs_random["centers"].q_distribution()
        else:
            q_centers = self.inputs_nonrandom["centers"]

        std = positive_exp(shape=self.shape)
        return GMMClustering(weights=q_weights, centers=q_centers, std=std, shape=self.shape, name="q_"+self.name)

    def _hack_symmetry_correction(self):
        permutation_correction = np.sum(np.log(np.arange(1, self.n_clusters+1))) # log (n_centers)!
        return permutation_correction
        
    
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

    def derived_parameters(self, B, G, std, **kwargs):
        derived = {}
        derived["variance"] = std**2
        derived["mean"] = tf.matmul(B, G)
        return derived
    
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
        bernoulli_params = q_B.p
        
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
        # not implemented yet. we have to return a value, but
        # let's make sure it's never used.
        return tf.constant(np.nan)

    def default_q(self):
        return Gaussian(shape=self.shape, name="q_"+self.name)
    
    def _hack_symmetry_correction(self):
        permutation_correction = np.sum(np.log(np.arange(1, self.K+1))) # log K!
        return permutation_correction
    
class MultiplicativeGaussianNoise(ConditionalDistribution):    

    def __init__(self, A, std, **kwargs):
        super(MultiplicativeGaussianNoise, self).__init__(A=A, std=std, **kwargs)   

    def inputs(self):
        return {"A": unconstrained, "std": positive_exp}

    def _compute_shape(self, A_shape, std_shape):
        return A_shape
    
    def _sample(self, A, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        noise = eps * std
        return A * noise

    def _logp(self, result, A, std):
        residuals = result / A
        lp = tf.reduce_sum(util.dists.gaussian_log_density(residuals, 0.0, std))
        return lp

