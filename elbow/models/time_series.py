import numpy as np
import tensorflow as tf

import elbow.util as util
from elbow import ConditionalDistribution

import scipy.stats

from elbow.gaussian_messages import MVGaussianMeanCov, reverse_message, forward_message

from elbow.parameterization import unconstrained, psd_matrix, psd_diagonal

class LinearGaussian(ConditionalDistribution):
    
    def __init__(self, shape, K, prior_mean, prior_cov,
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

        self.T, self.D = shape
        self.K = K
        
        if observation_mean is None:
            self._flag_no_obs = True
            observation_mean = tf.zeros((self.D,), dtype=tf.float32)
            observation_mat = tf.constant(np.float32(np.eye(self.D)))
            observation_cov = tf.zeros((self.D, self.D), dtype=tf.float32)
        
        super(LinearGaussian, self).__init__(prior_mean=prior_mean, prior_cov=prior_cov,
                                             transition_mat=transition_mat,
                                             transition_mean=transition_mean,
                                             transition_cov=transition_cov,
                                             observation_mat=observation_mat,
                                             observation_mean=observation_mean,
                                             observation_cov=observation_cov,
                                             shape=shape, **kwargs)


    def inputs(self):
        return {"prior_mean": unconstrained, "prior_cov": psd_matrix,
                 "transition_mat": unconstrained, "transition_mean": unconstrained, "transition_cov": psd_matrix,
                 "observation_mat": unconstrained, "observation_mean": unconstrained, "observation_cov": psd_matrix}


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

    def _sample_and_entropy(self, **input_samples):
        sampled = self._sample(**input_samples)
        entropy = -self._logp(result=sampled, **input_samples)
        return sampled, entropy
    
    def _sample(self, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat=None, observation_mean=None, observation_cov=None):

        transition_mean = tf.reshape(transition_mean, shape=(self.D, 1))
        transition_eps = tf.random_normal(shape=(self.T, self.D))
        transition_epses = [tf.reshape(n, shape=(self.D, 1)) for n in tf.unpack(transition_eps)]

        prior_mean = tf.reshape(prior_mean, shape=(self.D, 1))
        prior_cov_L = tf.cholesky(prior_cov)
        state = prior_mean + tf.matmul(prior_cov_L, transition_epses[0])

        transition_cov_L = tf.cholesky(transition_cov)
        if not self._flag_no_obs:
            observation_cov_L = tf.cholesky(observation_cov)
            obs_eps = tf.random_normal(shape=(self.T, self.K))
            obs_epses = [tf.reshape(n, shape=(self.K, 1)) for n in tf.unpack(obs_eps)]
            observation_mean = tf.reshape(observation_mean, shape=(self.K, 1))
            
        output = []
        hidden = []
        for t in range(self.T):
            if not self._flag_no_obs:
                pred_obs = tf.matmul(observation_mat, state) + observation_mean
                output.append(pred_obs + tf.matmul(observation_cov_L, obs_epses[t]))
            else:
                output.append(state)
            hidden.append(state)

            if t < self.T-1:
                state_noise = transition_mean + tf.matmul(transition_cov_L, transition_epses[t+1])
                state = tf.matmul(transition_mat, state) + state_noise

        self._sampled_hidden = hidden
        return tf.pack(tf.squeeze(output))
    
    def _logp(self, result, prior_mean, prior_cov,
                 transition_mat, transition_mean, transition_cov,
                 observation_mat=None, observation_mean=None, observation_cov=None):
    
        # define the Kalman filtering calculation within the TF graph
        if not self._flag_no_obs:
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

            if not self._flag_no_obs:

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
                
            step_logp = util.dists.multivariate_gaussian_log_density(y, 0, S)

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
                 step_noise_means,
                 step_noise_covs,
                 unary_means,
                 unary_variances, **kwargs):

        super(LinearGaussianChainCRF, self).__init__(transition_matrices=transition_matrices, step_noise_means=step_noise_means, step_noise_covs=step_noise_covs, unary_means=unary_means, unary_variances=unary_variances, shape=shape, **kwargs)


    def inputs(self):
        return {"transition_matrices": unconstrained, "step_noise_means": unconstrained, "step_noise_covs": psd_diagonal, "unary_means": None, "unary_variances": None}

    def _sample_and_entropy(self, transition_matrices,
                            step_noise_means,
                            step_noise_covs,
                            unary_means,
                            unary_variances):

        T, d = self.shape
        
        upwards_means = tf.unpack(unary_means)
        upwards_vars = tf.unpack(unary_variances)
        unary_factors = [MVGaussianMeanCov(mean, tf.diag(vs)) for (mean, vs) in zip(upwards_means, upwards_vars)]

        # transition_matrices is either a d x d matrix, or a T x d x d tensor
        if len(transition_matrices.get_shape()) == 2:
            transition_matrices = [transition_matrices for i in range(T)]

        # step noise mean is either a (d,)-vector or a T x d matrix
        if len(step_noise_means.get_shape()) == 1:
            step_noise_means = [step_noise_means for i in range(T)]

        # step noise cov is either a d x d matrix or a T x d x d tensor
        if len(step_noise_covs.get_shape()) == 2:
            step_noise_covs = [step_noise_covs for i in range(T)]

        step_noise_factors = [MVGaussianMeanCov(step_noise_means[t], step_noise_covs[t]) for t in range(T)]
            
        back_filtered, logZ = self._pass_messages_backwards(transition_matrices,
                                                            step_noise_factors,
                                                            unary_factors)

        self._back_filtered = back_filtered
        self._logZ = logZ
        
        eps = tf.random_normal(shape=self.shape)
        sample, entropy = self._sample_forward(back_filtered, transition_matrices,
                                               step_noise_factors, eps)
        return sample, entropy

    def _entropy(self):
        raise Exception("can't compute entropy without a sample...")
        
    def _sample(self):
        raise Exception("shouldn't try to sample a chainCRF without entropy...")
        
    def _pass_messages_backwards(self, transition_matrices, step_noise_factors, unary_factors):
        messages = []
        T, d = self.shape
        
        back_filtered = unary_factors[T-1]
        messages.append(back_filtered)
        logZ = 0.0
        for t in np.arange(T-1)[::-1]:
            back_filtered_pred = reverse_message(back_filtered,
                                                 transition_matrices[t],
                                                 step_noise_factors[t])

            logZ += back_filtered_pred.multiply_density_logZ(unary_factors[t])
            back_filtered = back_filtered_pred.multiply_density(unary_factors[t])

            messages.append(back_filtered)

        messages = messages[::-1]
        return messages, logZ

    def _sample_forward(self, back_filtered, transition_matrices,
                        step_noise_factors, eps):
        samples = []

        T, d = self.shape
        epses = tf.unpack(eps)

        sampling_dist = back_filtered[0]
        z_i = sampling_dist.sample(epses[0])
        samples.append(z_i)

        sampling_dists = [sampling_dist]        
        entropies = [sampling_dist.entropy()]
        for t in np.arange(1, T):
            pred_mean = tf.matmul(transition_matrices[t-1], z_i)
            noise = step_noise_factors[t-1]

            incoming = MVGaussianMeanCov(noise.mean() + pred_mean, noise.cov())
            
            sampling_dist = back_filtered[t].multiply_density(incoming)
            sampling_dists.append(sampling_dist)
            
            z_i = sampling_dist.sample(epses[t])
            entropies.append(sampling_dist.entropy())
            samples.append(z_i)

        self.sampling_dists = sampling_dists
        self.entropies = entropies

        entropy = tf.reduce_sum(tf.pack(entropies))
        sample = tf.reshape(tf.squeeze(tf.pack(samples)), self.shape)
        return sample, entropy
