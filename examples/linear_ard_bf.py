import numpy as np
import tensorflow as tf

import bayesflow as bf

from util import batch_generator

# TODO:
#  - figure out the story with transforms and initialization. I should really be able to specify a transform and easily compute its inverse.
#  - check that generated density (and MAP solution) matches raw impl
#  - check that generated ELBO matches raw implementation 
#  - test against Stan

def linear_ard_joint_density(batch_X, batch_y, alpha, sigma2, w, N):

    a0 = 1.0
    b0 = 1.0
    c0 = 1.0
    d0 = 1.0
    
    batch_size = batch_y.get_shape()[0].value

    alpha_prior = bf.dists.gamma_log_density(alpha, alpha=a0, beta=b0)
    sigma2_prior = bf.dists.inv_gamma_log_density(sigma2, alpha=c0, beta=d0)
    w_prior = bf.dists.gaussian_log_density(w, mean=0, stddev=tf.sqrt(sigma2/alpha))

    ybar = tf.matmul(batch_X,tf.expand_dims(w, 1))
    y_lik = bf.dists.gaussian_log_density(batch_y,
                                          mean=ybar,
                                          stddev=tf.sqrt(sigma2))

    
    joint_density = y_lik + float(batch_size)/N * alpha_prior + sigma2_prior + w_prior

    return joint_density

def simulate(X, alpha, sigma2):
    N, d = X.shape
    w_stddev = np.sqrt(sigma2) * 1.0/np.sqrt(alpha)
    w = np.random.randn(d) * w_stddev
    f = np.dot(X, w)
    y = f + np.random.randn() * np.sqrt(sigma2)
    
    return w, y


def main():  

    
    N = 10000
    d = 250
    alpha = np.ones((d),)
    alpha[d/2:] = 10.0
    sigma2 = 1.0
    X = np.random.rand(N, d)
    w, y = simulate(X, alpha, sigma2)

    batch_size = 64
    batch_X = tf.placeholder(tf.float32, (batch_size, d), name="X")
    batch_y = tf.placeholder(tf.float32, (batch_size, ), name="y")

    mf = bf.mean_field.MeanFieldInference(linear_ard_joint_density, 
                                          batch_X=batch_X, 
                                          batch_y=batch_y,
                                          N=N)

    a0 = 1.0
    b0 = 1.0
    c0 = 1.0
    d0 = 1.0
    
    alpha_default = np.ones((d,), dtype=np.float32) * a0/b0
    mf.add_latent("alpha", 
                  1/np.sqrt(alpha_default), 
                  1e-6 * np.ones((d,), dtype=np.float32), 
                  bf.transforms.exp_reciprocal,
                  shape=(d,))
    sigma2_default = np.array(d0/(c0+1)).astype(np.float32)
    mf.add_latent("sigma2", 
                  np.sqrt(sigma2_default), 
                  1e-6, 
                  bf.transforms.square,
                  shape=())
    mf.add_latent("w", 
                  tf.random_normal([d,], stddev=1.0, dtype=tf.float32),
                  1e-6 * np.ones((d,), dtype=np.float32),
                  shape=(d,))
    

    
    elbo = mf.build_stochastic_elbo(n_eps=5)
    sigma2s = mf.get_posterior_samples("sigma2")
    #alphas = mf.get_posterior_samples("alpha")
    alpha_mean_var = mf.latents["alpha"]["q_mean"]
    alpha_stddev_var = mf.latents["alpha"]["q_stddev"]
    alpha_var = mf.latents["alpha"]["samples"][0]
    
    train_step = tf.train.AdamOptimizer(0.01).minimize(-elbo)
    debug = tf.add_check_numerics_ops()
    init = tf.initialize_all_variables()
    merged = tf.merge_all_summaries()
    
    sess = tf.Session()
    writer = tf.train.SummaryWriter("/tmp/ard_logs", sess.graph_def)
    sess.run(init)
    
    for i, batch_xs, batch_ys in batch_generator(X, y, 64, max_steps=20000):
        fd = mf.sample_stochastic_inputs()
        fd[batch_X] = batch_xs
        fd[batch_y] = batch_ys

        (elbo_val, sigma2s_val, alpha_mean, alpha_stddev, alpha_val) = sess.run([elbo, sigma2s, alpha_mean_var, alpha_stddev_var, alpha_var], feed_dict=fd)
        
        print "step %d elbo %.2f sigma2 %.2f " % (i, elbo_val, np.mean(sigma2s_val))

        summary_str = sess.run(merged, feed_dict=fd)
        writer.add_summary(summary_str, i)


        try:
            sess.run(debug, feed_dict=fd)
        except:
            bad = ~np.isfinite(alpha_val)
            print alpha_mean[bad]
            print alpha_stddev[bad]
            print alpha_val[bad]
            
        sess.run(train_step, feed_dict = fd)


        #if i % 50 ==0:
        #(elbo_val, sigma2s_val, alpha_val) = sess.run([elbo, sigma2s, alphas], feed_dict=fd)
        #print "step %d elbo %.2f sigma2 %.2f" % (i, elbo_val, np.mean(sigma2s_val))


if __name__ == "__main__":
    main()
