import numpy as np
import tensorflow as tf

import bayesflow as bf
import scipy.stats

"""
Feature learning with an Indian Buffet Process (IBP) type model. 
Specifically we model the observed matrix X as the product of a latent 
feature indicator matrix Z with a matrix A of feature values, plus Gaussian noise. 

Formally, X is an N x D matrix, with K latent features (the full IBP allows 
   infinite latent features...)
Z: N x K, specifying whether feature k is active in observation n.
   Entries in each row are Bernoulli ~ pi, where the feature probabilities pi 
   are drawn from a Beta prior (taken to be Beta(1,1) = uniform in this example).

A: K x D, gives the representations of the latent features, with entries drawn 
   from a Gaussian prior. 

Taking a mean-field Gaussian posterior representation for q(A), and a mean-field 
Bernoulli representation for q(Z), the expectations in the ELBO are all computable 
analytically, so we do not actually require the reparameterization trick or
other Monte Carlo methods. 
"""

def sample_synthetic(N, D, K, sigma_a = 1.0, sigma_n=0.1, seed=0):

    np.random.seed(seed)

    pi = scipy.stats.beta(1,1).rvs(K)
    
    Z = np.random.rand(N, K) < pi
    A = np.random.randn(K, D) * sigma_a

    X = np.dot(Z, A) + np.random.randn(N, D) * sigma_n

    return X, A, Z, pi
    

def expected_log_likelihood(bernoulli_params, q_A_means, q_A_stds, X_means, X_stds, noise_stds):
    """ 
    compute E_Q{X, A, Z} [ log p(X | A, Z) ]
      where q(A) ~ N(q_A_means, q_A_stds)
            q(Z) ~ Bernoulli(bernoulli_params)
            q(X) ~ N(q_X_means, q_X_stds)
                   (a posterior on X representing an upwards message.
                   In the object-level case the variances are zero)
      and the model itself is
          log p(X | A, Z) ~ N(X; ZA, noise_stds)
      i.e. each entry of X is modeled as a Gaussian with mean given 
      by the corresponding entry of ZA, and stddev given by the 
      corresponding entry of noise_stds. (in principle these can all
      be different though in practice we will usually have a single
      global stddev, or perhaps per-column or per-row stddevs). 

    
    Matrix shapes: N datapoints, D dimensions, K latent features
     X: N x D
     A: K x D
     Z: N x K
    """

    #N, D = util._tf_extract_shape(X_means)
    #K, D2 = util._tf_extract_shape(q_A_means)
    #N2, K2 = util._tf_extract_shape(bernoulli_params)
    #assert(D==D2)
    #assert(N==N2)
    #assert(K==K2)

    
    expected_X = tf.matmul(bernoulli_params, q_A_means)
    effective_var = tf.square(X_stds) + tf.square(noise_stds)
    precisions = 1.0/effective_var
    gaussian_lp = bf.dists.gaussian_log_density(X_means, expected_X, variance=effective_var)
    
    mu2 = tf.square(q_A_means)
    tau_V = tf.matmul(bernoulli_params, tf.square(q_A_stds))
    tau_tau2_mu2 = tf.matmul(bernoulli_params - tf.square(bernoulli_params), mu2)
    tmp = tau_V + tau_tau2_mu2
    lp_correction = tmp * precisions
    
    pointwise_expected_lp = gaussian_lp - .5*lp_correction 
    expected_lp = tf.reduce_sum(pointwise_expected_lp)

    return expected_lp


def main():

    K = 3
    D = 10
    N = 100

    sigma_n = 0.3
    sigma_a = 1.0
    
    X, A, Z, pi = sample_synthetic(N, D, K, sigma_a, sigma_n)

    A = np.float32(A)
    X = np.float32(X)
    
    # take the bernoulli prior as fixed for simplicity
    prior_probs = np.float32(np.tile(pi, (N, 1)))

    pz = tf.constant(prior_probs)
    # parameterize the bernoulli probs as log-odds.
    # the log jacobian is irrelevant here because this isn't a transform of a model variable. 
    qz_raw = tf.Variable(np.float32(np.zeros((N, K))), name="qz_raw")
    qz, log_jacobian = bf.transforms.logit(qz_raw)
    
    init_mean = np.float32(np.random.randn(K, D))
    init_stddev = np.float32(np.ones((K, D)) * 1e-4)
    q_A_means = tf.Variable(init_mean, name="q_A_mean")
    q_A_logstddev = tf.Variable(np.log(init_stddev), name="q_A_stddev")
    q_A_stddev, log_jacobian = bf.transforms.exp(q_A_logstddev)

    p_A_means = tf.constant(np.float32(np.zeros((K, D))))
    p_A_stddevs = tf.constant(np.float32(np.ones((K, D)) * sigma_a ))

    # treat X as fully observed
    x_mean = tf.constant(X)
    x_stds = tf.constant(np.float32(np.zeros((N, D))))

    with tf.name_scope("ell") as scope:
        ell = expected_log_likelihood(qz, q_A_means, q_A_stddev, x_mean, x_stds, noise_stds=sigma_n)
    kl_bernoulli = tf.reduce_sum(bf.dists.bernoulli_kl(qz, pz))
    kl_gaussian = tf.reduce_sum(bf.dists.gaussian_kl(q_A_means, tf.square(q_A_stddev), p_A_means, tf.square(p_A_stddevs)))
        
    with tf.name_scope("elbo") as scope:
        elbo = ell - kl_bernoulli - kl_gaussian 


    
    train_step = tf.train.AdamOptimizer(0.1).minimize(-elbo)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        sess.run(train_step)
        elbo_val, ell_val, kl_gaussian_val, kl_bernoulli_val = sess.run((elbo, ell, kl_gaussian, kl_bernoulli))
        qz_val, qAm, qAs = sess.run((qz, q_A_means, q_A_stddev))
        print i, elbo_val, ell_val, kl_gaussian_val, kl_bernoulli_val

        qX = np.dot(qz_val, qAm)
        print "reconstruction error", np.linalg.norm(qX-X)

    print "initial reconstruction error", np.linalg.norm(np.dot(prior_probs, init_mean)-X)
    print "true reconstruction error", np.linalg.norm(np.dot(Z, A)-X)

        
if __name__ == "__main__":
    main()
