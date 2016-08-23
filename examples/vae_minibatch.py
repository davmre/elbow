import numpy as np
import tensorflow as tf

import bayesflow as bf

from bayesflow import Model
from bayesflow.elementary import Gaussian
from bayesflow.models.neural import NeuralGaussian, NeuralBernoulli

from util import mnist_training_data

import time

def build_vae(d_z=2, d_hidden=256, d_x=784, batchsize=100, batch_scaling=1.0):

    # MODEL
    z = Gaussian(mean=0, std=1.0, shape=(batchsize,d_z), name="z", minibatch_scale_factor=batch_scaling)
    X = NeuralBernoulli(z, d_hidden=d_hidden, d_x=d_x, name="X", minibatch_scale_factor=batch_scaling)

    # OBSERVED DATA
    x_batch = tf.placeholder(shape=(batchsize, d_x), dtype=tf.float32)
    X.observe(x_batch)

    # VARIATIONAL POSTERIOR
    q_z = NeuralGaussian(x_batch, d_hidden=d_hidden, d_z=d_z, name="q_z")
    z.attach_q(q_z)

    jm = Model(X)
    
    return jm, x_batch

def main():
    batchsize = 100
    batch_scaling = 60000./batchsize
    jm, x_batch = build_vae(batchsize=batchsize, batch_scaling=batch_scaling)

    elbo = jm.construct_elbo()
    
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
            fd = {}
            fd[x_batch] = Xtrain[start:end]

            sess.run(train_step, feed_dict = fd)
            elbo_val = sess.run((elbo), feed_dict=fd)            
            print start, elbo_val

        elapsed = time.time() - tstart
        print "epoch", i_epoch, "time", elapsed

if __name__ == "__main__":
    main()
