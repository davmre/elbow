import numpy as np
import tensorflow as tf

import elbow.util as util

from elbow.conditional_dist import ConditionalDistribution
from elbow.parameterization import unconstrained, positive_exp, simplex_constrained, unit_interval
from elbow.transforms import Logit, Simplex, Exp, TransformedDistribution, Normalize
from elbow.elementary import Gaussian

import scipy.stats

FIX_TRIANGLE, FIX_IDENTITY, FIX_NONE = np.arange(3)

class MaskedGaussian(Gaussian):
    """
    Gaussian Q distribution that fixes some set of masked entries to be zero, and 
    ignores these entries when computing entropy, logp, etc. 
    """
    
    def __init__(self, mean=None, std=None, fix=FIX_TRIANGLE, shape=None, **kwargs):

        m, k = shape
        self.fix = fix
        if fix==FIX_IDENTITY:
            mask = np.asarray(np.vstack((np.eye(k), np.ones((m-k, k)))), dtype=np.float32)
        elif fix==FIX_TRIANGLE:
            mask = np.asarray(np.tril(np.ones((m, k))), dtype=np.float32)
        else:
            self.n_fixed = 0
            mask = np.ones((m, k), dtype=np.float32)
        self.mask = tf.constant(mask)
        self.flat_mask = tf.constant(np.asarray(mask, dtype=np.bool).flatten())
        
        super(MaskedGaussian, self).__init__(mean=mean, std=std, shape=shape, **kwargs)

        self.mean = self.mask * self.mean
        self.std = self.mask * self.std
        self.variance = tf.square(self.std)
        
    def inputs(self):
        return {"mean": unconstrained, "std": positive_exp}

    def outputs(self):
        return ("out",)
    
    def _sample(self, mean, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)        
        return self.mask * (eps * std + mean)
    
    def _logp(self, result, mean, std):

        rf = self.flatten_and_mask(result)
        mf = self.flatten_and_mask(mean)
        sf = self.flatten_and_mask(std)
        
        lp = tf.reduce_sum(util.dists.gaussian_log_density(rf, mean=mf, stddev=sf))
        return lp

    def _entropy(self, std, **kwargs):
        variance = self.flatten_and_mask(std**2)
        return tf.reduce_sum(util.dists.gaussian_entropy(variance=variance))

    def default_q(self, **kwargs):
        raise NotImplementedError

    def flatten_and_mask(self, t):
        # given a tensor of shape self.shape, flatten it and remove
        # any entries that are zeroed by self.mask. This avoids things
        # like computing the entropy of a zero-variance Gaussian.
        flat = tf.reshape(t, [-1])
        return tf.boolean_mask(flat, self.flat_mask)
    
    def _expected_logp(self, q_result, q_mean=None, q_std=None):

        def get_sample(q, param):
            return self.inputs_nonrandom[param] if q is None else q._sampled

        
        std_sample = self.flatten_and_mask(get_sample(q_std, 'std'))
        mean_sample = self.flatten_and_mask(get_sample(q_mean, 'mean'))
        result_sample = self.flatten_and_mask(q_result._sampled)
        
        if isinstance(q_result, Gaussian) and not isinstance(q_mean, Gaussian):
            rm = self.flatten_and_mask(q_result.mean)
            rv = self.flatten_and_mask(q_result.variance)
            cross = util.dists.gaussian_cross_entropy(rm, rv, mean_sample, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        elif not isinstance(q_result, Gaussian) and isinstance(q_mean, Gaussian):
            mm = self.flatten_and_mask(q_mean.mean)
            mv = self.flatten_and_mask(q_mean.variance)
            cross = util.dists.gaussian_cross_entropy(mm, mv, result_sample, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        elif isinstance(q_result, Gaussian) and isinstance(q_mean, Gaussian):
            rm = self.flatten_and_mask(q_result.mean)
            rv = self.flatten_and_mask(q_result.variance)
            mm = self.flatten_and_mask(q_mean.mean)
            mv = self.flatten_and_mask(q_mean.variance)

            cross = util.dists.gaussian_cross_entropy(mm, mv+rv, rm, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        else:
            result_sample = q_result._sampled
            std_sample = get_sample(q_std, 'std')
            mean_sample = get_sample(q_mean, 'mean')
            elp = self._logp(result=result_sample, mean=mean_sample, std=std_sample)
        return elp
            
    def _compute_shape(self, mean_shape, std_shape):
        raise NotImplementedError

    def _input_shape(self, param, **kwargs):
        assert (param in self.inputs().keys())
        return self.shape
    
    def reparameterized(self):
        return True



    
class SignFlipGaussian(Gaussian):

    def __init__(self, deg=10, **kwargs):
        self.deg = deg
        self.x, self.w = np.polynomial.hermite.hermgauss(deg)

        super(SignFlipGaussian, self).__init__(**kwargs)
    
    def _entropy(self, mean, std, **kwargs):
        n, k = self.shape
        
        col_means = tf.split(1, k, mean)
        col_stds = tf.split(1, k, std)

        trait_entropies = []

        alpha2s = []
        
        for i, (m, s) in enumerate(zip(col_means, col_stds)):

            with tf.name_scope("entropy_col_%d" % i):
                v = s**2

                entropy = util.dists.gaussian_entropy(variance=v)

                # now compute the approximate correction

                # compute alpha2 = mu^T Sigma^-1 mu for the multivariate Gaussian with 
                # mu = [m]
                # Sigma  = diag([v])
                alpha2 = tf.reduce_sum(tf.square(m) / v) 

                alpha2 = tf.clip_by_value(alpha2, 1e-10, 1e8)
                
                alpha2s.append(alpha2)
                
                # y ~ N(mean=alpha2, var=alpha2) are the Monte Carlo
                # "variables", except here we construct them from
                # the deterministic quadrature points x
                y = tf.mul(tf.sqrt(alpha2),self.x) + alpha2
                fs = -tf.log(.5*(1 + tf.exp(-2*y)))

                # gauss-hermite quadrature is for integrals wrt e(-x^2); 
                # adapting this to a Gaussian density requires a change of
                # variables introducing a 1/sqrt(pi) factor. 
                correction = tf.reduce_sum(tf.mul(fs, self.w)) / np.sqrt(np.pi)

                trait_entropies.append(entropy + correction)

        self.alpha2s = tf.pack(alpha2s)
                
        return tf.add_n(trait_entropies)


class GaussianMonteCarlo(Gaussian):
    def _entropy(self, *args, **kwargs):
        return -self._parameterized_logp(*args, result=self._sampled, **kwargs)


class ExplicitPermutationMixture(Gaussian):
    
    def __init__(self, deg=10, **kwargs):
        super(ExplicitPermutationMixture, self).__init__(**kwargs)

    def _entropy(self, mean, std, **kwargs):
        return -self._parameterized_logp(result=self._sampled, **kwargs)

    def _logp(self, result, mean, std, **kwargs):
        import itertools

        n, k = self.shape
        col_means = tf.unpack(mean, axis=1)
        col_stds = tf.unpack(std, axis=1)

        components = zip(col_means, col_stds)

        
        component_lps = []
        self.perm_mean_list = []
        for perm in itertools.permutations(components):
            perm_means, perm_stds = zip(*perm)
            perm_mean = tf.pack(perm_means, axis=1)
            perm_std = tf.pack(perm_stds, axis=1)

            print "perm", perm
            self.perm_mean_list.append(perm_mean)
            
            lp = tf.reduce_sum(util.dists.gaussian_log_density(result, mean=perm_mean, stddev=perm_std))
            component_lps.append(lp)

        component_lps = tf.pack(component_lps)

        self.component_lps = component_lps
        
        max_lp  = tf.reduce_max(component_lps)
        lp = tf.log(tf.reduce_mean(tf.exp(component_lps - max_lp))) + max_lp
        return lp


def lpbessel_svs(xs, n):
    # compute a Laplace approximation (Butler & Wood, 2002)
    # to the matrix-argument Bessel function
    #   int_T exp( UT ) V(dT)
    # for a matrix U with V(dT) representing 
    # Haar measure on the orthogonal group T \in O(n)
    #
    # xs: singular values of U'U

    def r(u):
        return u/(tf.sqrt(tf.square(u) + 1.0) + 1.0)
    
    us = 2.0*xs / n
    ys = r(us)
    y2 = tf.square(ys)
    y2r = tf.reshape(y2, (n, 1))
    y2pairs = tf.matmul( y2r, tf.transpose(y2r))
    log_denom = .5 * tf.reduce_sum(tf.log(1-y2pairs))
    log_num = tf.reduce_sum(xs*ys + n/2.0 * tf.log((1-y2)))
    return log_num - log_denom

class DiagonalRotationMixture(Gaussian):
    """
    posterior approximation for an elementwise matrix Gaussian
    density symmetrized with respect to the orthogonal group O(n)
    acting in the column space.
    
    Currently assumes covariance is isotropic. This can be enforced by constructing
      std = tf.tile( tf.exp(tf.Variable(np.asarray(-2).reshape(1,1), dtype=tf.float32)), (n,k))
      qdist = DiagonalRotationMixture(..., shape=(n, k), std=std)
    where the std is a scalar tiled to the full shape (n,k).
    """    

    def __init__(self, **kwargs):
        super(DiagonalRotationMixture, self).__init__(**kwargs)
        
    def _entropy(self, mean, std, **kwargs):
        return -self._parameterized_logp(result=self._sampled, **kwargs)

    def _logp(self, result, mean, std, **kwargs):

        n, k = self.shape
        base_logp = tf.reduce_sum(util.dists.gaussian_log_density(result, mean=mean, stddev=std))
        cxu = tf.matmul(tf.transpose(result/std), mean/std)
        svs = tf.sqrt(util.differentiable_sq_singular_vals(cxu))    
        lb = lpbessel_svs(svs, k)
        lp = base_logp + lb - tf.trace(cxu)
        
        return lp
