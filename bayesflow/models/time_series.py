import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution

import scipy.stats

class LinearGaussian(ConditionalDistribution):
    
    def __init__(self, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat, observation_mean, observation_cov, T,
                 **kwargs):
        # a linear Gaussian state-space model (aka Kalman filter) with dimensions
        # T: number of time steps
        # D: dimension of hidden state
        # K: dimension of output
        self.T = T
        
        super(LinearGaussian, self).__init__(prior_mean=prior_mean, prior_cov=prior_cov,
                                             transition_mat=transition_mat,
                                             transition_mean=transition_mean,
                                             transition_cov=transition_cov,
                                             observation_mat=observation_mat,
                                             observation_mean=observation_mean,
                                             observation_cov=observation_cov,  **kwargs)

        _, self.K = self.output_shape
        self.D = self.input_nodes["prior_mean"].output_shape[0]

    def inputs(self):
        return ("prior_mean", "prior_cov",
                 "transition_mat", "transition_mean", "transition_cov",
                 "observation_mat", "observation_mean", "observation_cov")
        
    def _compute_shape(self, prior_mean_shape, prior_cov_shape,
                       transition_mat_shape, transition_mean_shape, transition_cov_shape,
                       observation_mat_shape, observation_mean_shape, observation_cov_shape):

        D1, = prior_mean_shape
        D2, D3 = prior_cov_shape
        D4, D5 = transition_mat_shape
        D6, = transition_mean_shape
        D7, D8 = transition_cov_shape
        K1, D9 = observation_mat_shape
        K2, = observation_mean_shape
        K3, K4 = observation_cov_shape
        
        assert(D1 == D2 == D3 == D4 == D5 == D6 == D7 == D8 == D9)
        assert(K1 == K2 == K3 == K4)
        return (self.T, K1)
    
    def _compute_dtype(self, prior_mean_dtype, prior_cov_dtype,
                       transition_mat_dtype, transition_mean_dtype, transition_cov_dtype,
                       observation_mat_dtype, observation_mean_dtype, observation_cov_dtype):

        assert(prior_mean_dtype == prior_cov_dtype == transition_mat_dtype == transition_mean_dtype == transition_cov_dtype == observation_mat_dtype == observation_mean_dtype == observation_cov_dtype)
        return prior_mean_dtype
        
    def _sample(self, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat, observation_mean, observation_cov):

        prior = scipy.stats.multivariate_normal(mean=prior_mean, cov=prior_cov)
        state = prior.rvs(1)

        transition_noise = scipy.stats.multivariate_normal(mean=transition_mean, cov=transition_cov)
        observation_noise = scipy.stats.multivariate_normal(mean=observation_mean, cov=observation_cov)

        output = np.empty(self.output_shape, dtype=self.dtype)
        hidden = np.empty((self.T, self.D), dtype=self.dtype)
        for t in range(self.T):
            pred_obs = np.dot(observation_mat, state)
            output[t, :] = pred_obs + observation_noise.rvs(1)
            hidden[t, :] = state
            
            state = np.dot(transition_mat, state) + transition_noise.rvs(1)

        self._sampled_hidden = hidden
        return output
    
    def _logp(self, result, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat, observation_mean, observation_cov):
    
        # define the Kalman filtering calculation within the TF graph
        observation_mean = tf.reshape(observation_mean, (self.K, 1))
        transition_mean = tf.reshape(transition_mean, (self.D, 1))
        
        pred_mean = tf.reshape(prior_mean, (self.D, 1))
        pred_cov = prior_cov

        filtered_means = []
        filtered_covs = []
        step_logps = []
        observations = tf.unpack(result)
        for t in range(self.T):
            # TODO optimize this to not use an explicit matrix inverse

            tmp = tf.matmul(observation_mat, pred_cov)
            S = tf.matmul(tmp, tf.transpose(observation_mat)) + observation_cov
            Sinv = tf.matrix_inverse(S)
            gain = tf.matmul(pred_cov, tf.matmul(tf.transpose(observation_mat), Sinv))

            obs_t = tf.reshape(observations[t], (self.K, 1))
            y = obs_t - tf.matmul(observation_mat, pred_mean) - observation_mean
            updated_mean = pred_mean + tf.matmul(gain, y)
            updated_cov = pred_cov - tf.matmul(gain, tmp)
            #step_logp = bf.dists.multivariate_gaussian_log_density(y, 0, S)
            step_logp = bf.dists.multivariate_gaussian_log_density(y, 0, S)

            filtered_means.append(updated_mean)
            filtered_covs.append(updated_cov)
            step_logps.append(step_logp)

            if t < self.T-1:
                pred_mean = tf.matmul(transition_mat, updated_mean) + transition_mean
                pred_cov = tf.matmul(transition_mat, tf.matmul(updated_cov, tf.transpose(transition_mat))) + transition_cov

        self.filtered_means = filtered_means
        self.filtered_covs = filtered_covs
        self.step_logps = tf.pack(step_logps)
        logp = tf.reduce_sum(self.step_logps)

        return logp

