import tensorflow as tf
import numpy as np

from elbow import ConditionalDistribution

from elbow.parameterization import unconstrained, unconstrained_zeros, positive_exp, simplex_constrained, unit_interval, unconstrained_small, unconstrained_scale
from elbow.transforms import Logit, Simplex, Exp, TransformedDistribution, Normalize

from elbow.util.dists import multivariate_gaussian_log_density, multivariate_gaussian_entropy, gaussian_entropy, gaussian_log_density, bernoulli_entropy, bernoulli_log_density
from elbow.util.misc import concrete_shape

class NoisyRandomProjection(ConditionalDistribution):
    """
    Projects an input matrix Z through a random linear projection W,
    sampled from W ~ N(0, 1), marginalizing out the projection. 
    This is 'dual PCA' (ordinarily in PCA we marginalize out the inputs Z
    and solve for W). 
    """
    
    def __init__(self, Z, mu=None, std=None,  **kwargs):
        super(NoisyRandomProjection, self).__init__(Z=Z, mu=mu, std=std, **kwargs)
        
    def inputs(self):
        return {"Z": unconstrained, "mu": unconstrained, "std": positive_exp}    
    
    def _compute_shape(self, Z_shape, mu_shape, std_shape):
        raise Exception("RandomProjection requires an explicitly specified shape")
    
    def _sample(self, Z, mu, std):
        n, d_output = self.shape
        d_latent = concrete_shape(Z.get_shape())[1]
        W = tf.random_normal(shape=(d_output, d_latent), dtype=self.dtype)
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        return tf.matmul(Z, tf.transpose(W)) + mu + eps*std
    
    def _logp(self, result, Z, mu, std):
        n, d_output = self.shape
        
        cov = tf.matmul(Z, tf.transpose(Z)) + tf.diag(tf.ones(n,)*std) 
        L = tf.cholesky(cov)
        r = result - mu
        out_cols = tf.unpack(r, axis=1)
        lps = [multivariate_gaussian_log_density(col, mu=0., L=L) for col in out_cols]
        return tf.reduce_sum(tf.pack(lps))
    
    def _entropy(self, Z, mu, std):
        n, d_output = self.shape
        cov = tf.matmul(Z, tf.transpose(Z)) + tf.diag(tf.ones(n,)*std) 
        L = tf.cholesky(cov)
        return d_output * multivariate_gaussian_entropy(L=L)


class InverseProjection(ConditionalDistribution):
    """
    Define an 'inference network' performing exact posterior inference 
    in PCA. Given high-dimensional observations X and a projection matrix W,
    computes the (analytic) multivariate Gaussian posterior on the 
    latent variables X, assuming a standard Gaussian prior.

    Note that when constructing this RV as a variational model, the inputs
    should be the *variational models* of X and W (q_X and q_W) rather than
    X and W themselves. 
    """
    
    def __init__(self, X, W, mu=None, std=None, **kwargs):
        super(InverseProjection, self).__init__(X=X, W=W, mu=mu, std=std, **kwargs)
        
    def inputs(self):
        return {"X": None, "W": None, "mu": unconstrained, "std": positive_exp}
    
    def _build_inverse_projection(self, X, W, mu, std):
        n, d_latent = self.shape
        variance = tf.square(std)

        M = tf.matmul(tf.transpose(W), W) + tf.diag(tf.ones((d_latent))*variance)
        L = tf.cholesky(M)
        
        # pred_Z = M^-1 W' r' as per (6) in tipping & bishop JRSS
        # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ppca-jrss.pdf
        r = X-mu
        WTr = tf.transpose(tf.matmul(r, W)) # W' (t - mu)  
        tmp = tf.matrix_triangular_solve(L, WTr, lower=True)
        pred_z = tf.transpose(tf.matrix_triangular_solve(tf.transpose(L), tmp, lower=False))        
        
        return pred_z, L, std
        
    def _sample_and_entropy(self, **input_samples):
        # more efficient to re-use the same network for sampled value and entropy
        pred_z, L, std = self._build_inverse_projection(**input_samples)
        eps = tf.random_normal(shape=self.shape)
        unit_entropy = multivariate_gaussian_entropy(L_prec=L/std)
        
        self.canonical_pred = pred_z
        self.canonical_L = L        

        n, d_latent = self.shape
        sample = pred_z + tf.transpose(tf.matrix_triangular_solve(tf.transpose(L/std), tf.transpose(eps), lower=False))     
        entropy = n*unit_entropy        
        return sample, entropy
        
    def _sample(self, X, W, mu, std):
        pred_z, L, std = self._build_inverse_projection(X, W, mu, std)
        eps = tf.random_normal(shape=self.shape)
        
        # sample from N(pred_z, M^-1 * std**2)
        # where LL' = M
        return pred_z + tf.transpose(tf.matrix_triangular_solve(tf.transpose(L/std), tf.transpose(eps), lower=False))     
            
    def _entropy(self, X, W, mu, std):
        n, d_latent = self.shape
        pred_z, L, std = self._build_inverse_projection(X, W, mu, std)
        unit_entropy = multivariate_gaussian_entropy(L_prec=L/std)
        return n*unit_entropy
    
    def _logp(self, result, X, W, mu, std):
        pred_z, L, std = self._build_inverse_projection(X, W, mu, std)
        
        rows = tf.unpack(result - pred_z)
        lps = [multivariate_gaussian_log_density(r, mu=0, L_prec=L/std) for r in rows]
        return tf.reduce_sum(tf.pack(lps))


class MeanFieldLinearGaussian(ConditionalDistribution):
    
    def __init__(self, X, W=None, mu=None, std=None, **kwargs):
        self.X_shape = concrete_shape(Z.get_shape())[1] if isinstance(X, tf.Tensor) else X.shape
        super(MeanFieldLinearGaussian, self).__init__(X=X, W=W, mu=mu, std=std, **kwargs)
        
    def inputs(self):
        return {"X": None, "W": unconstrained, "mu": unconstrained_zeros, "std": positive_exp}

    def _compute_shape(self, X_shape, W_shape, mu_shape, std_shape):
        n, d = X_shape
        m, d = W_shape
        return (n, m)
    
    def _input_shape(self, param, **kwargs):
        assert(self.shape is not None)
        n, m = self.shape
        n2, d = self.X_shape
        assert(n==n2)
        if param=="W":
            return (m, d)
        elif param=="mu":
            return (m,)
        elif param=="std":
            return (m,)
        else:
            raise Exception("unknown param %s" % param)
            
    def _sample(self, X, W, mu, std):
        
        # X: n x d
        # W: m x d
        # Z: n x m
        proj = tf.matmul(X, tf.transpose(W)) + mu         
        eps = tf.random_normal(shape=self.shape)
        sampled = proj + eps*std
        return sampled
    
    def _entropy(self, X, W, mu, std):
        n, m = self.shape
        return n*gaussian_entropy(stddev=std)
        
    
    def _logp(self, result, X, W, mu, std):
        proj = tf.matmul(X, tf.transpose(W)) + mu    
        return gaussian_log_density(result, mu=proj, stddev=std)

    def derived_parameters(self, X, W, mu, std, **kwargs):
        proj = tf.matmul(X, tf.transpose(W)) + mu    
        return {"mean": proj}

    
class MeanFieldBernoulli(ConditionalDistribution):
    
    def __init__(self, X, W=None, b=None, **kwargs):
        self.X_shape = concrete_shape(Z.get_shape())[1] if isinstance(X, tf.Tensor) else X.shape
        self.scale = 10.0
        super(MeanFieldBernoulli, self).__init__(X=X, W=W, b=b, **kwargs)


        
    def inputs(self):
        return {"X": None, "W": unconstrained, "b": unconstrained_small}

    def _compute_shape(self, X_shape, W_shape, mu_shape, std_shape):
        n, d = X_shape
        m, d = W_shape
        return (n, m)
    
    def _input_shape(self, param, **kwargs):
        assert(self.shape is not None)
        n, m = self.shape
        n2, d = self.X_shape
        assert(n==n2)
        if param=="W":
            return (m, d)
        elif param=="b":
            return (m,)
        else:
            raise Exception("unknown param %s" % param)
            
    def _sample(self, X, W, b):
        
        # X: n x d
        # W: m x d
        # Z: n x m
        proj = tf.matmul(X, tf.transpose(W)) + b
        probs = Logit.transform(proj / self.scale)

        unif = tf.random_uniform(shape=self.shape, dtype=tf.float32)
        return tf.cast(unif < probs, self.dtype)
    
    def _entropy(self, X, W, b):
        n, m = self.shape
        proj = tf.matmul(X, tf.transpose(W)) + b
        probs = Logit.transform(proj / self.scale)
        return tf.reduce_sum(bernoulli_entropy(probs))
        
    def _logp(self, result, X, W, b):
        proj = tf.matmul(X, tf.transpose(W)) + b
        probs = Logit.transform(proj / self.scale)
        lps = bernoulli_log_density(result, probs)
        return tf.reduce_sum(lps)

    def derived_parameters(self, X, W, b, **kwargs):
        proj = tf.matmul(X, tf.transpose(W)) + b
        probs = Logit.transform(proj / self.scale)
        return {"p": probs}
