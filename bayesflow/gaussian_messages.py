
import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

"""
WARNING: untested, known correctness bugs
"""

class AbstractMVGaussian(object):

    def __init__(self, d):
        self.d = d

    def log_p(self, x):
        raise Exception("not implemented")

    def entropy(self):
        return self._entropy

    def mean(self):
        return self._mean

    def cov(self):
        return self._cov

    def prec_mean(self):
        return self._prec_mean

    def prec(self):
        return self._prec

    def entropy(self):
        return self._entropy

    def sample(self, eps):        
        return tf.matmul(self._L_cov, tf.reshape(eps, (-1, 1)))
    
    def multiply_density(self, other_gaussian):
        # the product of two Gaussian densities has
        # the form of an unnormalized Gaussian density,
        # with normalizing constant given by the
        # method multiply_density_logZ.
        assert(self.d == other_gaussian.d)
        new_prec = self.prec() + other_gaussian.prec()
        new_prec_mean = self.prec_mean() + other_gaussian.prec_mean()
        return MVGaussianNatural(new_prec_mean, new_prec, d=self.d)

    def multiply_density_logZ(self, other_gaussian):
        assert(self.d == other_gaussian.d)
        tmp = self.subtract(other_gaussian)
        return tmp.log_p(tf.zeros(shape=(self.d,)))
    
    def add(self, other_gaussian):
        assert(self.d == other_gaussian.d)
        
        new_mean = self.mean() + other_gaussian.mean()
        new_cov = self.cov() + other_gaussian.cov()

        return MVGaussianMeanCov(new_mean, new_cov, d=self.d)

    def subtract(self, other_gaussian):
        assert(self.d == other_gaussian.d)
        
        new_mean = self.mean() - other_gaussian.mean()
        new_cov = self.cov() + other_gaussian.cov()

        return MVGaussianMeanCov(new_mean, new_cov, d=self.d)

    def linear_transform(self, A):
        n, m = util.extract_shape(A)
        assert(m == self.d)
        new_mean = tf.matmul(A, self.mean)
        new_cov = tf.matmul(tf.matmul(A, self.cov), tf.transpose(A))
        return MVGaussianMeanCov(new_mean, new_cov, d=n)

    def inverse_linear_transform(self, A):
        # treat this as a distribution on Ax, and
        # unpack the (implied) distribution on x
        m, n = util.extract_shape(A)
        assert(m == self.d)

        At = tf.transpose(A)
        new_prec = tf.matmul(At, tf.matmul(self.prec(), A))
        new_prec_mean = tf.matmul(At, self.prec_mean())
        return MVGaussianNatural(new_prec_mean, new_prec, d=n)
    
    def condition(self, A, y, sigma_obs):
        # return posterior from observing y = N(Ax, sigma_obs)
        raise Exception("not implemented")

class MVGaussianMeanCov(AbstractMVGaussian):

    def __init__(self, mean, cov, d=None):

        try:
            d1, = util.extract_shape(mean)
            mean = tf.reshape(mean, (d1,1))
        except:
            d1,k = util.extract_shape(mean)
            assert(k == 1)

        d2,_ = util.extract_shape(cov)
        assert(d1==d2)
        if d is None:
            d = d1
        else:
            assert(d==d1)
            
        super(MVGaussianMeanCov, self).__init__(d=d)
        
        self._mean = mean
        self._cov = cov
        
        self._L_cov = tf.cholesky(cov)
        self._entropy = bf.dists.multivariate_gaussian_entropy(L=self._L_cov)

        L_prec_transpose = util.triangular_inv(self._L_cov)
        self._L_prec = tf.transpose(L_prec_transpose)
        self._prec = tf.matmul(self._L_prec, L_prec_transpose)
        self._prec_mean = tf.matmul(self._prec, self._mean)

    def log_p(self, x):
        x_flat = tf.reshape(x, (-1,))
        mean_flat = tf.reshape(self._mean, (-1,))
        return bf.dists.multivariate_gaussian_log_density(x_flat, mean_flat, L=self._L_cov)
    
class MVGaussianNatural(AbstractMVGaussian):

    def __init__(self, prec_mean, prec, d=None):

        try:
            d1, = util.extract_shape(prec_mean)
            prec_mean = tf.reshape(prec_mean, (d1,1))
        except:
            d1,k = util.extract_shape(prec_mean)
            assert(k == 1)

            
        d2,_ = util.extract_shape(prec)
        assert(d1==d2)
        if d is None:
            d = d1
        else:
            assert(d==d1)

        super(MVGaussianNatural, self).__init__(d=d)

        self._prec_mean = prec_mean
        self._prec = prec
        
        self._L_prec = tf.cholesky(prec)
        self._entropy = bf.dists.multivariate_gaussian_entropy(L_prec=self._L_prec)

        # want to solve prec * mean = prec_mean for mean.
        # this is equiv to (LL') * mean = prec_mean.
        # since tf doesn't have a cholSolve shortcut, just
        # do it directly:
        #   solve L y = prec_mean
        # to get y = (L' * mean), then
        #   solve L' mean = y
        y = tf.matrix_triangular_solve(self._L_prec, self._prec_mean, lower=True, adjoint=False)
        self._mean = tf.matrix_triangular_solve(self._L_prec, y, lower=True, adjoint=True)

        L_cov_transpose = util.triangular_inv(self._L_prec)
        self._L_cov = tf.transpose(L_cov_transpose)
        self._cov = tf.matmul(self._L_cov, L_cov_transpose)

    def log_p(self, x):
        x_flat = tf.reshape(x, (-1,))
        mean_flat = tf.reshape(self._mean, (-1,))

        return bf.dists.multivariate_gaussian_log_density_natural(x_flat, mu=mean_flat,
                                                                  prec=self._prec,
                                                                  L_prec=self._L_prec)


    
def reverse_message(gaussian_tplus1, transition_mat,
                    gaussian_noise):
    """
    Given a transition model
      N(x_{t+1}; T x_t + b; S)
    and future message
      N(x_{t+1}, c, C) 
    compute the message arriving at the current timestep
      N(x_t, d, D)
    where D^-1 = T' (C+S)^-1 T
          D^-1 d = T' (C+S)^-1 (c - b)
    """

    denoised = gaussian_tplus1.subtract(gaussian_noise)
    return denoised.inverse_linear_transform(transition_mat)
    
def forward_message(gaussian_t, transition_mat,
                    gaussian_noise=None):
    """
    Given a transition model
      N(x_{t+1}; T x_t + b; S)
    and current message
      N(x_t, c, C) 
    compute the message arriving at the next timestep
      N(x_{t+1}, d, D)
    where D = T C T' + S
          d = T c + b
    """

    if gaussian_noise is None:
        if noise_bias is None:
            noise_bias = tf.zeros_like(gaussian_t.mean())
        gaussian_noise = MVGaussianMeanCov(noise_bias, noise_cov)
    
    pred_gaussian = gaussian_t.linear_transform(transition_mat)
    noisy_gaussian = pred_gaussian.add(gaussian_noise)
    return noisy_gaussian

