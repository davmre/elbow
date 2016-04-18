import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import construct_elbo, FlatDistribution
from bayesflow.models.elementary import GaussianMatrix, BernoulliMatrix
from bayesflow.models.q_distributions import DeltaQDistribution, BernoulliQDistribution
from bayesflow.models.matrix_decompositions import *
from bayesflow.models.transforms import PointwiseTransformedMatrix
from bayesflow.models.neural import VAEEncoder, VAEDecoderBernoulli, init_weights, init_zero_vector

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

def clustering_gmm_model(n_clusters = 4,
                         cluster_center_std = 5.0,
                         cluster_spread_std = 2.0,
                         n_points = 500,
                         dim = 2):

    centers = GaussianMatrix(mean=0.0, std=cluster_center_std, output_shape=(n_clusters, dim))

    uniform_weights = np.float32(np.ones((n_clusters,)) / float(n_clusters))
    weights = FlatDistribution(value=uniform_weights, fixed=False)
    X = GMMClustering(weights=weights, centers=centers, std=cluster_spread_std, output_shape=(n_points, dim))

    sampled_X = X.sample(seed=0)
    X.observe(sampled_X)

    q_centers = centers.attach_gaussian_q()

    # TODO build a smarter approach for Q distributions in a transformed space
    tf_unnorm_weights = tf.Variable(np.float32(np.zeros((n_clusters,))), name="unnorm_weights")
    tf_weights, _ = bf.transforms.simplex(tf_unnorm_weights)
    q_weights = DeltaQDistribution(tf_weights)
    weights.attach_q(q_weights)


    elbo, sample_stochastic = construct_elbo(X)
    return elbo, sample_stochastic

def latent_feature_model():
    K = 3
    D = 10
    N = 100

    pi = np.float32(np.random.rand(K))

    B = BernoulliMatrix(p=pi, output_shape=(N, K))
    G = GaussianMatrix(mean=0.0, std=1.0, output_shape=(K, D))
    D = NoisyLatentFeatures(B=B, G=G, std=0.1)

    sampled_D = D.sample(seed=0)
    D.observe(sampled_D)

    q_G = G.attach_gaussian_q()
    q_B = BernoulliQDistribution(shape=(N, K))
    B.attach_q(q_B)

    elbo, sample_stochastic = construct_elbo(D)

    return elbo, sample_stochastic


def sparsity():
    G1 = GaussianMatrix(mean=0, std=1.0, output_shape=(100,10))
    expG1 = PointwiseTransformedMatrix(G1, bf.transforms.exp)
    X = MultiplicativeGaussianNoise(expG1, 1.0)

    sampled_X = X.sample()
    X.observe(sampled_X)

    q_G1 = G1.attach_gaussian_q()
    elbo, sample_stochastic = construct_elbo(X)
    
    return elbo, sample_stochastic

def autoencoder():
    d_z = 2
    d_hidden=256
    d_x = 28*28
    N=100

    from util import get_mnist
    Xdata, ydata = get_mnist()
    Xbatch = Xdata[0:N]
    
    def init_decoder_params(d_z, d_hidden, d_x):
        # TODO come up with a simpler/more elegant syntax for point weights.
        # maybe just let the decoder initialize and manage its own weights / q distributions? 
        w_decode_h = DeltaQDistribution(init_weights(d_z, d_hidden))
        w_decode_h2 = DeltaQDistribution(init_weights(d_hidden, d_x))
        b_decode_1 = DeltaQDistribution(init_zero_vector(d_hidden))
        b_decode_2 = DeltaQDistribution(init_zero_vector(d_x))
        
        w1 = FlatDistribution(value=np.zeros((d_z, d_hidden), dtype=np.float32), fixed=False)
        w1.attach_q(w_decode_h)
        w2 = FlatDistribution(value=np.zeros(( d_hidden, d_x), dtype=np.float32), fixed=False)
        w2.attach_q(w_decode_h2)
        b1 = FlatDistribution(value=np.zeros((d_hidden,), dtype=np.float32), fixed=False)
        b1.attach_q(b_decode_1)
        b2 = FlatDistribution(value=np.zeros((d_x), dtype=np.float32), fixed=False)
        b2.attach_q(b_decode_2)
        
        return w1, w2, b1, b2

    w1, w2, b1, b2 = init_decoder_params(d_z, d_hidden, d_x)
    z = GaussianMatrix(mean=0, std=1.0, output_shape=(N,d_z))
    X = VAEDecoderBernoulli(z, w1, w2, b1, b2)
    
    X.observe(Xbatch)
    tfX = tf.constant(Xbatch, dtype=tf.float32)

    q_z = VAEEncoder(tfX, d_hidden, d_z)
    z.attach_q(q_z)
    
    elbo, sample_stochastic = construct_elbo(X)
    return elbo, sample_stochastic

def toy_inference(elbo, sample_stochastic, adam_rate=0.2, steps=1000):
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
    
    print "gaussian mixture model"
    elbo, sample_stochastic = clustering_gmm_model()
    toy_inference(elbo, sample_stochastic)

    print "latent features"
    elbo, sample_stochastic = latent_feature_model()
    toy_inference(elbo, sample_stochastic)
    
    print "bayesian sparsity"
    elbo, sample_stochastic = sparsity()
    toy_inference(elbo, sample_stochastic)

    print "variational autoencoder"
    elbo, sample_stochastic = autoencoder()
    toy_inference(elbo, sample_stochastic, adam_rate=0.001)
    
    
if __name__ == "__main__":
    main()
