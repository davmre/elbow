import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import construct_elbo
from bayesflow.models.elementary import GaussianMatrix
from bayesflow.models.matrix_decompositions import NoisyGaussianMatrixProduct, NoisyCumulativeSum

def gaussian_mean_model():
    mu = GaussianMatrix(mean=0, std=10, output_shape=(1,))
    q_mu = mu.attach_gaussian_q()
    
    X = GaussianMatrix(mean=mu, std=1, output_shape=(100,))

    sampled_X = X.sample(seed=2)
    X.observe(sampled_X)

    elbo, sample_stochastic = construct_elbo(X)

    return elbo, sample_stochastic
    
def gaussian_lowrank_model():
    A = GaussianMatrix(mean=0.0, std=1.0, output_shape=(100, 3))
    B = GaussianMatrix(mean=0.0, std=1.0, output_shape=(100, 3))
    C = NoisyGaussianMatrixProduct(A=A, B=B, std=0.1)

    sampled_C = C.sample(seed=0)
    C.observe(sampled_C)

    q_A = A.attach_gaussian_q()
    q_B = B.attach_gaussian_q()
    elbo, sample_stochastic = construct_elbo(C)

    return elbo, sample_stochastic
    
def gaussian_randomwalk_model():
    A = GaussianMatrix(mean=0.0, std=1.0, output_shape=(100, 2))
    C = NoisyCumulativeSum(A=A, std=0.1)

    sampled_C = C.sample(seed=0)
    C.observe(sampled_C)

    q_A = A.attach_gaussian_q()
    elbo, sample_stochastic = construct_elbo(C)

    return elbo, sample_stochastic

def toy_inference(elbo, sample_stochastic, adam_rate=0.5, steps=1000):
    train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    for i in range(steps):
        fd = sample_stochastic()
        sess.run(train_step, feed_dict = fd)
        elbo_val = sess.run((elbo), feed_dict=fd)
        print i, elbo_val

def main():
    print "gaussian mean estimation"
    elbo, sample_stochastic = gaussian_mean_model()
    toy_inference(elbo, sample_stochastic)

    print "gaussian matrix factorization"
    elbo, sample_stochastic = gaussian_lowrank_model()
    toy_inference(elbo, sample_stochastic)

    print "gaussian random walk"
    elbo, sample_stochastic = gaussian_randomwalk_model()
    toy_inference(elbo, sample_stochastic)

if __name__ == "__main__":
    main()
