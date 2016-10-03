import numpy as np
import tensorflow as tf
import elbow.util.dists as dists
import scipy.stats

"""
Bernoulli example model using a score fn gradient estimator
instead of the reparameterization trick.

Mostly a proof of concept, convergence is much slower than the
reparameterization trick. The advantage is that we are not limited to
a (transformed) Gaussian posterior; here we can choose a Beta
posterior which is (by conjugacy) actually the true form. Of course in
practice you should just do the exact conjugate update and not bother
with noisy variational inference. :-)

"""


class BernoulliModel(object):

    def __init__(self, N, n_thetas=1):

        self.N = N

        self.theta_q_alpha = tf.Variable(1.0, name="theta_q_alpha")
        self.theta_q_beta = tf.Variable(2.0, name="theta_q_beta")

        self.data = tf.placeholder(dtype=tf.float32, shape=(N,), name="data")

        self.thetas = tf.placeholder(shape=(n_thetas,), dtype=tf.float32, name="thetas")
        
        self.thetas_q_log_density = tf.reduce_sum(dists.beta_log_density(self.thetas, alpha=self.theta_q_alpha, beta=self.theta_q_beta))
        self.thetas_prior = tf.reduce_sum(dists.beta_log_density(self.thetas, alpha=1., beta=1.) )

        self.data_liks = tf.pack([tf.reduce_sum(dists.bernoulli_log_density(self.data, theta)) for theta in tf.unpack(self.thetas)])
        self.joint_density = self.data_liks + self.thetas_prior
        
        self.stochastic_elbo = self.joint_density - self.thetas_q_log_density

        # TODO: add control variates
        self.surrogate = tf.reduce_mean(self.thetas_q_log_density * tf.stop_gradient(self.stochastic_elbo) + self.stochastic_elbo)




bernoulli_data = (0,1,0,0,0,0,0,0,0,1)
N = len(bernoulli_data)

n_thetas = 100
model = BernoulliModel(N, n_thetas=n_thetas)
    
train_step = tf.train.AdamOptimizer(0.1).minimize(-model.surrogate)
init = tf.initialize_all_variables()
        
sess = tf.Session()
sess.run(init)

for i in range(10000):
    feed_dict = {model.data: bernoulli_data}

    alpha, beta = sess.run([model.theta_q_alpha, model.theta_q_beta])
    
    rv = scipy.stats.beta(alpha, beta)
    thetas = rv.rvs(n_thetas)
    feed_dict[model.thetas] = thetas

    (q, prior, lik, joint, elbos, surrogate) = sess.run([model.thetas_q_log_density, model.thetas_prior, model.data_liks, model.joint_density, model.stochastic_elbo, model.surrogate, ], feed_dict=feed_dict)

    if i % 100 == 0:
        print "step %d alpha %.3f beta %.3f theta %.3f q_density %.2f prior %.2f lik %.2f joint %.2f  elbo %.2f surrogate %.2f " % (i, alpha, beta, np.mean(thetas), np.mean(q), np.mean(prior), np.mean(lik), np.mean(joint), np.mean(elbos), surrogate)

    sess.run(train_step, feed_dict = feed_dict)
        
