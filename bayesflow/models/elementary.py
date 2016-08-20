import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution, JMContext, current_scope
from bayesflow.parameterization import unconstrained, positive_exp, simplex_constrained, unit_interval
from bayesflow.transforms import normalize
from bayesflow.models.transforms import PointwiseTransformedMatrix

import scipy.stats

class GammaMatrix(ConditionalDistribution):
    
    def __init__(self, alpha, beta, **kwargs):
        super(GammaMatrix, self).__init__(alpha=alpha, beta=beta, **kwargs)
    
    def inputs(self):
        return {"alpha": positive_exp, "beta": positive_exp}
    
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

    def _default_variational_model(self):
        q1 = Gaussian(shape=self.shape, model=None)
        return PointwiseTransformedMatrix(q1, bf.transforms.exp, model=None)

    def reparameterized(self):
        return False
    
class BetaMatrix(ConditionalDistribution):
    
    def __init__(self, alpha, beta, **kwargs):
        super(BetaMatrix, self).__init__(alpha=alpha, beta=beta, **kwargs)
    
    def inputs(self):
        return {"alpha": positive_exp, "beta": positive_exp}
    
    def _sample(self, alpha, beta):
        X = tf.random_gamma(shape=self.shape, alpha=alpha, beta=1)
        Y = tf.random_gamma(shape=self.shape, alpha=beta, beta=1)
        Z = X/(X+Y)
        return Z

    def _logp(self, result, alpha, beta):    
        lp = tf.reduce_sum(bf.dists.beta_log_density(result, alpha, beta))
        return lp

    def _compute_shape(self, alpha_shape, beta_shape):
        return bf.util.broadcast_shape(alpha_shape, beta_shape)
        
    def _compute_dtype(self, alpha_dtype, beta_dtype):
        assert(alpha_dtype==beta_dtype)
        return alpha_dtype

    def _default_variational_model(self):
        q1 = Gaussian(shape=self.shape, model=None)
        return PointwiseTransformedMatrix(q1, bf.transforms.logit, model=None)

    def reparameterized(self):
        return False



        
class DirichletMatrix(ConditionalDistribution):
    """
    Currently just describes a vector of shape (K,), though could be extended to 
    a (N, K) matrix of iid draws. 
    """
    
    def __init__(self, alpha, **kwargs):
        super(DirichletMatrix, self).__init__(alpha=alpha, **kwargs)
        self.K = self.shape[0]
        
    def inputs(self):
        return {"alpha": positive_exp}
    
    def _sample(self, alpha):

        # broadcast alpha from scalar to vector, if necessary
        alpha = alpha * tf.ones(shape=(self.shape[0]), dtype=self.dtype)

        gammas = tf.squeeze(tf.random_gamma(shape=(1,), alpha=alpha, beta=1))
        sample, log_jacobian = normalize(gammas)
        return sample

    def _logp(self, result, alpha):    
        lp = tf.reduce_sum(bf.dists.dirichlet_log_density(result, alpha))
        return lp

    def _compute_shape(self, alpha_shape):        
        return alpha_shape
        
    def _compute_dtype(self, alpha_dtype):
        return alpha_dtype

    def _default_variational_model(self):
        return SimplexTransformedGaussian(shape=(self.K,), name="q_"+self.name, model=None)
    
    def reparameterized(self):
        return False
    
class BernoulliMatrix(ConditionalDistribution):
    def __init__(self, p=None, **kwargs):
        super(BernoulliMatrix, self).__init__(p=p, **kwargs)        

        self.probs = self.input_val("p")
        
    def inputs(self):
        return {"p": unit_interval}

    def _input_shape(self, param):
        assert (param in self.inputs().keys())
        return self.shape
                     
    def _sample(self, p):
        unif = tf.random_uniform(shape=self.shape, dtype=tf.float32)
        return tf.cast(unif < p, self.dtype)
    
    def _expected_logp(self, q_result, q_p):
        # compute E_q [log p(z)] for a given prior p(z) and an approximate posterior q(z).
        # note q_p represents our posterior uncertainty over the parameters p: if these are known and
        # fixed, q_p is just a delta function, otherwise we have to do Monte Carlo sampling.
        # Whereas q_z represents our posterior over the sampled (Bernoulli) values themselves,
        # and we assume this is in the form of a set of Bernoulli probabilities. 

        p_z = q_p._sampled
        q_z = q_result.probs
        
        lp = -tf.reduce_sum(bf.dists.bernoulli_entropy(q_z, cross_q = p_z))
        return lp

    def _entropy(self, p):
        return tf.reduce_sum(bf.dists.bernoulli_entropy(p))
    
    def _compute_shape(self, p_shape):
        return p_shape
        
    def _compute_dtype(self, p_dtype):
        return np.int32

    def _default_variational_model(self):
        return BernoulliMatrix(shape=self.shape, model=None)
    
    def reparameterized(self):
        return False
    
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
        N, K = self.shape        
        choices = tf.multinomial(tf.log(p), num_samples=N)

        """
        M = np.zeros(N, K, dtype=self.dtype)
        r = np.arange(N)
        M[r, choices] = 1
        """

        M = tf.one_hot(choices, depth=K, axis=-1)
        return M
    
    def _expected_logp(self, q_result, q_p):
        p = q_p._sampled
        q = q_result.probs

        lp = tf.reduce_sum(bf.dists.multinomial_entropy(q, cross_q=p))
        return lp
    
    def _compute_shape(self, p_shape):
        return p_shape
        
    def _compute_dtype(self, p_dtype):
        return np.int32

    def reparameterized(self):
        return False



def transformed_distribution(base_dist, transform):

    class TransformedDistribution(ConditionalDistribution):
        
        def __init__(self, shape, **kwargs):

            self.base = base_dist(shape=shape, model=None)
            super(TransformedDistribution, self).__init__(**kwargs)

        def inputs(self):
            return self.base.inputs()
            
        def _compute_shape(self, *args, **kwargs):
            base_shape = self.base.shape
            # TODO implement this with transforms that can govern shape properly...
            #transformed_shape = 

        def _sample(self, **kwargs):
            pass
        
        def _logp(self, **kwargs):
            pass

        def _expected_logp(self, **kwargs):
            pass

        def _entropy(self, **kwargs):
            pass

    return TransformedDistribution
#SimplexTransformedGaussian = transformed_distribution(Gaussian, padded_simplex)

class SimplexTransformedGaussian(ConditionalDistribution):
    # temporary hack all around. need to figure out the generic form for a deterministic transform. I think this requires:
    #   - transform objects that known their inverses and effects on shape
    #   - convention of storing a cached (monte carlo) entropy estimate at each node along with the sampled value. 
    def __init__(self, shape, mean=None, std=None, **kwargs):

        assert(shape is not None and len(shape)==1)
        n = shape[0]

        self.gaussian = Gaussian(mean=mean, std=std, shape=(n-1,), model=None)
        super(SimplexTransformedGaussian, self).__init__(shape=shape, mean=mean,
                                                         std=std, **kwargs)

    def inputs(self):
        return self.gaussian.inputs()

    def _input_shape(self, *args, **kwargs):
        return self.gaussian._input_shape(*args, **kwargs)
    
    def _compute_shape(self, *args, **kwargs):
        base_shape = self.base.shape
        # TODO implement this with transforms that can govern shape properly...
        #transformed_shape = 

    def _sample(self, **kwargs):
        sampled_gaussian = self.gaussian._sample(**kwargs)
        _simplex_input = tf.concat(0, [sampled_gaussian, tf.zeros((1,), dtype=tf.float32)])
        sample, log_jacobian = bf.transforms.simplex(_simplex_input)

        # MASSIVE HACK
        self._sampled_log_jacobian = log_jacobian
        return sample

    def _logp(self, **kwargs):
        raise Exception("not implemented")

    def _expected_logp(self, **kwargs):
        raise Exception("not implemented")

    def _entropy(self, **kwargs):
        # ALSO A HACK
        sampled_entropy = self.gaussian._sampled_entropy + self._sampled_log_jacobian
        return sampled_entropy


class Gaussian(ConditionalDistribution):
    
    def __init__(self, mean=None, std=None, **kwargs):

        super(Gaussian, self).__init__(mean=mean, std=std, **kwargs) 


        self.mean = self.input_val("mean")
        self.std = self.input_val("std")
        self.variance = tf.square(self.std)

    def inputs(self):
        return {"mean": unconstrained, "std": positive_exp}

    def outputs(self):
        return ("out",)
    
    def _sample(self, mean, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)        
        return eps * std + mean
    
    def _logp(self, result, mean, std):
        lp = tf.reduce_sum(bf.dists.gaussian_log_density(result, mean=mean, stddev=std))
        return lp

    def _entropy(self, std, **kwargs):
        variance = std**2
        return tf.reduce_sum(bf.dists.gaussian_entropy(variance=variance))

    def _default_variational_model(self):
        return Gaussian(shape=self.shape, name="q_"+self.name, model=None)

    def _expected_logp(self, q_result, q_mean=None, q_std=None):

        def get_sample(q, param):
            return self.inputs_nonrandom[param] if q is None else q._sampled
            
        std_sample = get_sample(q_std, 'std')
        mean_sample = get_sample(q_mean, 'mean')
        out_sample = q_result._sampled
        
        if isinstance(q_result, Gaussian) and not isinstance(q_mean, Gaussian):
            cross = bf.dists.gaussian_cross_entropy(q_result.mean, q_result.variance, mean_sample, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        elif not isinstance(q_result, Gaussian) and isinstance(q_mean, Gaussian):
            cross = bf.dists.gaussian_cross_entropy(q_mean.mean, q_mean.variance, out_sample, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        elif isinstance(q_result, Gaussian) and isinstance(q_mean, Gaussian):
            cross = bf.dists.gaussian_cross_entropy(q_mean.mean, q_mean.variance + q_result.variance, q_result.mean, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        else:
            elp = self._logp(out=out_sample, mean=mean_sample, std=std_sample)
        return elp
            
    def _compute_shape(self, mean_shape, std_shape):
        return bf.util.broadcast_shape(mean_shape, std_shape)

    def _input_shape(self, param):
        assert (param in self.inputs().keys())
        return self.shape
    
    def _compute_dtype(self, mean_dtype, std_dtype):
        assert(mean_dtype==std_dtype)
        return mean_dtype
        
    def reparameterized(self):
        return True
