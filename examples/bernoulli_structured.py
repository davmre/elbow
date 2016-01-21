import numpy as np
import tensorflow as tf
import bayesflow as bf
from bayesflow.mean_field import MeanFieldInference

"""

Bernoulli example from Stan: infer theta from observations under a Beta(1,1) prior.

"""

def bernoulli_joint_density(theta, data):        
    theta_prior = bf.dists.beta_log_density(theta, alpha=1, beta=1) 
    data_lik = tf.reduce_sum(bf.dists.bernoulli_log_density(data, theta))
    joint_density = data_lik + theta_prior
    return joint_density

def main():        
    bernoulli_data = tf.constant((0,1,0,0,0,0,0,0,0,1), dtype=tf.float32)

    mf = MeanFieldInference(bernoulli_joint_density, data=bernoulli_data)
    mf.add_latent("theta", 0.5, 1e-6, bf.transforms.logit)
    elbo = mf.build_stochastic_elbo(n_eps=10)
    thetas = mf.get_posterior_samples("theta")
    
    train_step = tf.train.AdamOptimizer(0.1).minimize(-elbo)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(10000):
        fd = mf.sample_stochastic_inputs()
        sess.run(train_step, feed_dict = fd)

        if i % 100 == 0:
            elbo_val, theta_vals = sess.run([elbo, thetas], feed_dict=fd)
            print "step %d elbo %.2f theta mean %.2f" % (i, elbo_val, np.mean(theta_vals))


if __name__ == "__main__":
    main()
