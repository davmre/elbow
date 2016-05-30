import numpy as np
import tensorflow as tf

import bayesflow as bf


from bayesflow.models.train import construct_elbo
from bayesflow.models.elementary import GaussianMatrix

from bayesflow.models.neural import VAEEncoder, VAEDecoderBernoulli, init_weights, init_zero_vector
from bayesflow.models import FlatDistribution
from bayesflow.models.q_distributions import DeltaQDistribution

from util import mnist_training_data

import time

def decoder_params(d_z, d_hidden, d_x):
    w1 = GaussianMatrix(0.0, 1.0, name="w1", output_shape=(d_z, d_hidden))
    w1.q_distribution().initialize_to_value(tf.random_normal(w1.output_shape, stddev=0.01, dtype=tf.float32))

    w2 = GaussianMatrix(0.0, 1.0, name="w2", output_shape=(d_hidden, d_x))
    w2.q_distribution().initialize_to_value(tf.random_normal(w2.output_shape, stddev=0.01, dtype=tf.float32))

    b1 = GaussianMatrix(0.0, 1.0, name="b1", output_shape=(d_hidden,))
    b1.q_distribution().initialize_to_value(tf.zeros(shape=b1.output_shape))

    b2 = GaussianMatrix(0.0, 1.0, name="b2", output_shape=(d_x,))
    b2.q_distribution().initialize_to_value(tf.zeros(shape=b2.output_shape))
    return w1, w2, b1, b2

def build_vae(d_z=2, d_hidden=256, d_x=784, batchsize=100, batch_scaling=1.0):

    # MODEL
    w1, w2, b1, b2 = decoder_params(d_z, d_hidden, d_x)
    z = GaussianMatrix(mean=0, std=1.0, output_shape=(batchsize,d_z), name="z", minibatch_scale_factor=batch_scaling)
    X = VAEDecoderBernoulli(z, w1, w2, b1, b2, name="X", minibatch_scale_factor=batch_scaling)

    # OBSERVED DATA
    x_batch = tf.placeholder(shape=(batchsize, d_x), dtype=tf.float32)
    X.observe(x_batch)

    # VARIATIONAL POSTERIOR
    q_z = VAEEncoder(x_batch, d_hidden, d_z)
    z.attach_q(q_z)
    
    return X, x_batch

def main():
    batchsize = 100
    batch_scaling = 60000./batchsize
    X, x_batch = build_vae(batchsize=batchsize, batch_scaling=batch_scaling)
    elbo, sample_stochastic_inputs, decompose_elbo, inspect_posterior = construct_elbo(X)

    adam_rate = 0.01
    train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    Xtrain, _, _, _ = mnist_training_data()
    for i_epoch in xrange(10):
        tstart = time.time()
        for start in xrange(0, Xtrain.shape[0], batchsize):
            end = start+batchsize
            fd = sample_stochastic_inputs()
            fd[x_batch] = Xtrain[start:end]

            sess.run(train_step, feed_dict = fd)
            elbo_val = sess.run((elbo), feed_dict=fd)            
            print start, elbo_val

        elapsed = time.time() - tstart
        print "epoch", i_epoch, "time", elapsed

if __name__ == "__main__":
    main()
