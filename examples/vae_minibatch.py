import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models.train import construct_elbo
from bayesflow.models.elementary import GaussianMatrix

from bayesflow.models.neural import VAEEncoder, VAEDecoderBernoulli, init_weights, init_zero_vector
from bayesflow.models import FlatDistribution
from bayesflow.models.q_distributions import DeltaQDistribution

import time

def build_vae(batchsize=100, batch_scaling=1.0):
    d_z = 2
    d_hidden=256
    d_x = 28*28

    def init_decoder_params(d_z, d_hidden, d_x):
        # TODO come up with a simpler/more elegant syntax for point weights.
        # maybe just let the decoder initialize and manage its own weights / q distributions? 
        w_decode_h = DeltaQDistribution(init_weights(d_z, d_hidden))
        w_decode_h2 = DeltaQDistribution(init_weights(d_hidden, d_x))
        b_decode_1 = DeltaQDistribution(init_zero_vector(d_hidden))
        b_decode_2 = DeltaQDistribution(init_zero_vector(d_x))
        
        w1 = FlatDistribution(value=np.zeros((d_z, d_hidden), dtype=np.float32), fixed=False, name="w1")
        w1.attach_q(w_decode_h)
        w2 = FlatDistribution(value=np.zeros(( d_hidden, d_x), dtype=np.float32), fixed=False, name="w2")
        w2.attach_q(w_decode_h2)
        b1 = FlatDistribution(value=np.zeros((d_hidden,), dtype=np.float32), fixed=False, name="b1")
        b1.attach_q(b_decode_1)
        b2 = FlatDistribution(value=np.zeros((d_x), dtype=np.float32), fixed=False, name="b2")
        b2.attach_q(b_decode_2)
        
        return w1, w2, b1, b2

    w1, w2, b1, b2 = init_decoder_params(d_z, d_hidden, d_x)
    z = GaussianMatrix(mean=0, std=1.0, output_shape=(batchsize,d_z), name="z", minibatch_scale_factor=batch_scaling)
    X = VAEDecoderBernoulli(z, w1, w2, b1, b2, name="X", minibatch_scale_factor=batch_scaling)

    x_batch = tf.placeholder(shape=(batchsize, d_x), dtype=tf.float32)
    X.observe(x_batch)
    q_z = VAEEncoder(x_batch, d_hidden, d_z)
    z.attach_q(q_z)
    
    return X, x_batch

def main():
    from util import get_mnist
    Xdata, ydata = get_mnist()
    Xtrain = Xdata[0:60000]
    Xtest = Xdata[60000:70000]
    sortinds = np.random.permutation(60000)
    Xtrain = Xtrain[sortinds]

    batchsize = 100
    batch_scaling = 1.0 # 60000./batchsize
    X, x_batch = build_vae(batchsize=batchsize, batch_scaling=batch_scaling)
    
    elbo, sample_stochastic_inputs, decompose_elbo, inspect_posterior = construct_elbo(X)

    adam_rate = 0.01
    train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    preds = []

    epochs = 10
    for i_epoch in xrange(epochs):
        tstart = time.time()
        for start in xrange(0, Xtrain.shape[0], batchsize):
            end = start+batchsize
            Xt = Xtrain[start:end]

            fd = sample_stochastic_inputs()
            fd[x_batch] = Xt

            sess.run(train_step, feed_dict = fd)
            elbo_val = sess.run((elbo), feed_dict=fd)            

            print start, elbo_val

        elapsed = time.time() - tstart
        print "epoch", i_epoch, "time", elapsed

if __name__ == "__main__":
    main()
