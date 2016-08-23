import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution

import scipy.stats

class LinearGaussian(ConditionalDistribution):
    
    def __init__(self, T, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat=None, observation_mean=None, observation_cov=None,
                 **kwargs):
        # a linear Gaussian state-space model (aka Kalman filter) with dimensions
        #   T: number of time steps
        #   D: dimension of hidden state
        #   K: dimension of output
        # We allow K=0 (implied by observation_mat=None) in which case the model is 
        # just a Markov chain -- this can be useful together or in conjunction with
        # non-Gaussian (e.g., neural/VAE) observation models.
        
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
                       observation_mat_shape=None, observation_mean_shape=None, observation_cov_shape=None):

        D1, = prior_mean_shape
        D2, D3 = prior_cov_shape
        D4, D5 = transition_mat_shape
        D6, = transition_mean_shape
        D7, D8 = transition_cov_shape
        assert(D1 == D2 == D3 == D4 == D5 == D6 == D7 == D8)
        if observation_mat_shape is not None:
            K1, D9 = observation_mat_shape
            K2, = observation_mean_shape
            K3, K4 = observation_cov_shape
            assert(D9 == D1)
            assert(K1 == K2 == K3 == K4)
        else:
            K1 = D1

        return (self.T, K1)
    
    def _compute_dtype(self, prior_mean_dtype, prior_cov_dtype,
                       transition_mat_dtype, transition_mean_dtype, transition_cov_dtype,
                       observation_mat_dtype=None, observation_mean_dtype=None, observation_cov_dtype=None):

        assert(prior_mean_dtype == prior_cov_dtype == transition_mat_dtype == transition_mean_dtype == transition_cov_dtype)
        if observation_mat_dtype is not None:
            assert(prior_mean_dtype == observation_mat_dtype == observation_mean_dtype == observation_cov_dtype)
        return prior_mean_dtype
        
    def _sample(self, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat=None, observation_mean=None, observation_cov=None):

        prior = scipy.stats.multivariate_normal(mean=prior_mean, cov=prior_cov)
        state = prior.rvs(1)

        transition_noise = scipy.stats.multivariate_normal(mean=transition_mean, cov=transition_cov)
        if observation_mat is not None:
            observation_noise = scipy.stats.multivariate_normal(mean=observation_mean, cov=observation_cov)

        output = np.empty(self.output_shape, dtype=self.dtype)
        hidden = np.empty((self.T, self.D), dtype=self.dtype)
        for t in range(self.T):
            if observation_mat is not None:
                pred_obs = np.dot(observation_mat, state)
                output[t, :] = pred_obs + observation_noise.rvs(1)
            else:
                output[t, :] = state
            hidden[t, :] = state
            state = np.dot(transition_mat, state) + transition_noise.rvs(1)

        self._sampled_hidden = hidden
        return output
    
    def _logp(self, result, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat=None, observation_mean=None, observation_cov=None):
    
        # define the Kalman filtering calculation within the TF graph
        if observation_mean is not None:
            observation_mean = tf.reshape(observation_mean, (self.K, 1))
            
        transition_mean = tf.reshape(transition_mean, (self.D, 1))
        
        pred_mean = tf.reshape(prior_mean, (self.D, 1))
        pred_cov = prior_cov

        filtered_means = []
        filtered_covs = []
        step_logps = []
        observations = tf.unpack(result)
        for t in range(self.T):
            obs_t = tf.reshape(observations[t], (self.K, 1))

            if observation_mat is not None:

                tmp = tf.matmul(observation_mat, pred_cov)
                S = tf.matmul(tmp, tf.transpose(observation_mat)) + observation_cov
                # TODO optimize this to not use an explicit matrix inverse
                #Sinv = tf.matrix_inverse(S)
                #gain = tf.matmul(pred_cov, tf.matmul(tf.transpose(observation_mat), Sinv))

                # todo worth implementing cholsolve explicitly?
                gain = tf.matmul(pred_cov, tf.transpose(tf.matrix_solve(S, observation_mat)))
                
                y = obs_t - tf.matmul(observation_mat, pred_mean) - observation_mean
                updated_mean = pred_mean + tf.matmul(gain, y)
                updated_cov = pred_cov - tf.matmul(gain, tmp)
            else:
                updated_mean = obs_t
                updated_cov = tf.zeros_like(pred_cov)
                S = pred_cov
                y = obs_t - pred_mean
                
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


class LinearGaussianChainCRF(ConditionalDistribution):

    # TODO this has not been updated from the old-style QDistribution
    # so is currently totally broken
    
    def __init__(self, shape,
                 transition_matrices,
                 step_noise,
                 unary_factors=None):

        super(LinearGaussianChainCRF, self).__init__(shape=shape)
        T, d = shape

        self.T = T
        self._transition_matrices = transition_matrices
        self._step_noise = step_noise

        
        if unary_factors is not None:
            self._unary_factors = unary_factors
        else:
            unary_factors = []
            for t in range(T):                
                init_prec_mean = np.float32(np.random.randn((d,)) * 10)
                unary_factor_prec_mean = tf.Variable(init_prec_mean, name="unary_factor_prec_mean_%d" % t)
                unary_factor_prec = posdef_variable(d, init_log_diag=10)
                unary_factor = MVGaussianNatural(unary_factor_prec_mean, unary_factor_prec)
                unary_factors.append(unary_factor)
            self._unary_factors = unary_factors
            
        self.stochastic_eps = tf.placeholder(dtype=np.float32, shape=self.output_shape, name="eps")

        self._back_filtered, self._logZ = self._pass_messages_backwards()
        self.sample, self._entropy = self._sample_forward(self._back_filtered, self.stochastic_eps)

    def entropy(self):
        return self._entropy
        
    def sample_stochastic_inputs(self):
        sampled_eps = np.random.randn(*self.output_shape)
        return {self.stochastic_eps : sampled_eps}
        
    def _transition_mat(self, t):
        try:
            return tf.convert_to_tensor(self._transition_matrices[t])
        except:
            return tf.convert_to_tensor(self._transition_matrices)
        
    def _gaussian_noise(self, t):
        try:
            return self._step_noise[t]
        except:
            return self._step_noise

    def _pass_messages_backwards(self):
        messages = []
        back_filtered = self._unary_factors[self.T-1]
        messages.append(back_filtered)
        logZ = 0.0
        for t in np.arange(self.T-1)[::-1]:
            back_filtered_pred = reverse_message(back_filtered,
                                                 self._transition_mat(t),
                                                 self._gaussian_noise(t))

            unary_factor = self._unary_factors[t]
            logZ += back_filtered_pred.multiply_density_logZ(unary_factor)
            back_filtered = back_filtered_pred.multiply_density(unary_factor)

            messages.append(back_filtered)

        messages = messages[::-1]
        return messages, logZ

    def _sample_forward(self, back_filtered, eps):
        samples = []

        epses = tf.unpack(eps)

        sampling_dist = back_filtered[0]
        z_i = sampling_dist.sample(epses[0])
        samples.append(z_i)

        sampling_dists = [sampling_dist]        
        entropies = [sampling_dist.entropy()]
        for t in np.arange(1, self.T):
            pred_mean = tf.matmul(self._transition_mat(t-1), z_i)
            noise = self._gaussian_noise(t-1)

            #new_prec_mean = noise.prec_mean() + tf.matmul(noise.prec(), pred_mean)
            #incoming = MVGaussianNatural(new_prec_mean, noise.prec())
            incoming = MVGaussianMeanCov(noise.mean() + pred_mean, noise.cov())
            
            sampling_dist = back_filtered[t].multiply_density(incoming)
            sampling_dists.append(sampling_dist)
            
            z_i = sampling_dist.sample(epses[t])
            entropies.append(sampling_dist.entropy())            
            samples.append(z_i)

        self.sampling_dists = sampling_dists
        self.entropies = entropies

        entropy = tf.reduce_sum(tf.pack(entropies))
        sample = tf.reshape(tf.squeeze(tf.pack(samples)), self.output_shape)
        return sample, entropy
