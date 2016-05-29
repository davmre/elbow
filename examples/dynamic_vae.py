import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models.time_series import LinearGaussian
from bayesflow.models.train import construct_elbo
from bayesflow.models.elementary import GaussianMatrix
from bayesflow.models.q_distributions import LinearGaussianChainCRF
from bayesflow.gaussian_messages import MVGaussianMeanCov
from bayesflow.models.neural import VAEDecoderBernoulli, VAEEncoder
from bayesflow.models.minibatch import PackModels

def sample_toy_synthetic(d_x, T, noise_std):
    # not yet used: intended to replicate the toy test case from the SVAE paper
    x = np.zeros((T, d_x))
    idx = 0
    incr = 1
    for t in range(T):
        x[t, idx] = 1.0
        if idx + incr == d_x or (idx + incr) < 0:
            incr = -incr
        idx += incr

    x += np.random.randn(T, d_x) * noise_std
    return x

def decoder_params(d_z, d_hidden, d_x):
    w1 = GaussianMatrix(0.0, 1.0, name="w1", output_shape=(d_z, d_hidden))
    w2 = GaussianMatrix(0.0, 1.0, name="w2", output_shape=(d_hidden, d_x))
    b1 = GaussianMatrix(0.0, 0.0001, name="b1", output_shape=(d_hidden,))
    b2 = GaussianMatrix(0.0, 0.0001, name="b2", output_shape=(d_x,))
    return w1, w2, b1, b2

def dynamic_vae(T = 50, d_z = 1, d_hidden=2, d_x = 10, batchsize=2, N=10):

    minibatch_scale_factor = float(N)/batchsize
    
    # MODEL
    transition_mat = np.eye(d_z, dtype=np.float32) #GaussianMatrix(mean=0, std=1.0, output_shape=(D, D), name="transition")
    transition_bias = np.zeros((d_z,), dtype=np.float32)
    transition_cov = np.eye(d_z, dtype=np.float32)
    step_noise = MVGaussianMeanCov(transition_bias, transition_cov)

    w1, w2, b1, b2 = decoder_params(d_z, d_hidden, d_x)

    zis = []
    for i in range(batchsize):
        zi = LinearGaussian(T, transition_bias, transition_cov,
                            transition_mat, transition_bias, transition_cov,
                            name="z_%d" % i, minibatch_scale_factor=minibatch_scale_factor)
        zis.append(zi)

    z_batch = PackModels(zis) # shape B x T x d_z
    x = VAEDecoderBernoulli(z_batch, w1, w2, b1, b2, name="x", minibatch_scale_factor = minibatch_scale_factor)

    # SYNTHETIC OBSERVATION
    x_sampled = x.sample(0)
    q_x = x.observe(x_sampled)

    # INFERENCE MODEL
    upwards_messages = VAEEncoder(q_x.sample, d_hidden, d_z)
    tmat = tf.constant(transition_mat)
    item_upwards_means = tf.unpack(upwards_messages.mean)
    item_upwards_vars = tf.unpack(upwards_messages.variance)
    for i, (upwards_means, upwards_vars) in enumerate(zip(item_upwards_means, item_upwards_vars)):
        time_means = tf.unpack(upwards_means)
        time_vars = tf.unpack(upwards_vars)
        unary_factors = [MVGaussianMeanCov(mean, tf.diag(vs)) for (mean, vs) in zip(time_means, time_vars)]

        q_zi = LinearGaussianChainCRF((T, d_z), tmat, step_noise, unary_factors)
        zis[i].attach_q(q_zi)

    return x, zis, x_sampled

def dynamic_vae_single(T = 50, d_z = 1, d_hidden=2, d_x = 10):

    # MODEL
    transition_mat = np.eye(d_z, dtype=np.float32) #GaussianMatrix(mean=0, std=1.0, output_shape=(D, D), name="transition")
    transition_bias = np.zeros((d_z,), dtype=np.float32)
    transition_cov = np.eye(d_z, dtype=np.float32)
    step_noise = MVGaussianMeanCov(transition_bias, transition_cov)

    w1, w2, b1, b2 = decoder_params(d_z, d_hidden, d_x)

    z = LinearGaussian(T, transition_bias, transition_cov,
                                          transition_mat, transition_bias, transition_cov,
                                          name="z")
    x = VAEDecoderBernoulli(z, w1, w2, b1, b2, name="x")

    # SYNTHETIC OBSERVATION
    x_sampled = x.sample(0)
    q_x = x.observe(x_sampled)

    # INFERENCE MODEL
    upwards_messages = VAEEncoder(q_x.sample, d_hidden, d_z)
    upwards_means = tf.unpack(upwards_messages.mean)
    upwards_vars = tf.unpack(upwards_messages.variance)
    unary_factors = [MVGaussianMeanCov(mean, tf.diag(vs)) for (mean, vs) in zip(upwards_means, upwards_vars)]
    tmat = tf.constant(transition_mat)
    q_z = LinearGaussianChainCRF((T, d_z), tmat, step_noise, unary_factors)
    z.attach_q(q_z)

    return x, z, x_sampled
    

def main():
    #x, z, x_sampled = dynamic_vae_single(T = 50, d_z = 1, d_hidden=2, d_x = 10)
    x, zis, x_sampled = dynamic_vae(T = 20, d_z = 1, d_hidden=2, d_x = 10)
    
    elbo, sample_stochastic_inputs, decompose_elbo, inspect_posterior = construct_elbo(x)

    adam_rate = 0.1
    steps=800
    train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    preds = []
    for i in range(steps):
        fd = sample_stochastic_inputs()
        sess.run(train_step, feed_dict = fd)
        if i % 5 ==0:
            elbo_val = sess.run((elbo), feed_dict=fd)
            print i, elbo_val

if __name__ == "__main__":
    main()
