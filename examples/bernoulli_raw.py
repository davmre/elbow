import numpy as np
import tensorflow as tf
import bayesflow as bf

class BernoulliModel(object):

    def __init__(self, N, n_eps=1):

        self.N = N

        self.theta_q_mean = tf.Variable(0.5, name="theta_q_mean")
        self.theta_q_stddev = tf.Variable(1e-6, name="theta_q_stddev")
        self.theta_q_entropy = bf.dists.gaussian_entropy(stddev=self.theta_q_stddev)

        self.epss = []
        self.elbo = self.theta_q_entropy
        self.data = tf.placeholder(dtype=tf.float32, shape=(N,), name="data")
        for i in range(n_eps):
            theta_eps = tf.placeholder(dtype=tf.float32, name="theta_eps")
            self.epss.append(theta_eps)
            
            transformed_theta = theta_eps * self.theta_q_stddev + self.theta_q_mean
            theta, theta_log_jacobian = bf.transforms.logit(transformed_theta)
            theta_prior = bf.dists.beta_log_density(theta, alpha=1, beta=1) + theta_log_jacobian

            data_lik = bf.dists.bernoulli_log_density(self.data, theta)
            joint_density = data_lik + theta_prior
        
            self.elbo = self.elbo + 1.0/n_eps * joint_density
        
bernoulli_data = (0,1,0,0,0,0,0,0,0,1)
N = len(bernoulli_data)

model = BernoulliModel(N, n_eps=10)
    
train_step = tf.train.AdamOptimizer(0.1).minimize(-model.elbo)
init = tf.initialize_all_variables()
        
sess = tf.Session()
sess.run(init)

for i in range(10000):
    feed_dict = {model.data: bernoulli_data}
    for eps in model.epss:
        feed_dict[eps] = np.random.randn()
        
    sess.run(train_step, feed_dict = feed_dict)
        
    if i % 100 == 0:

        (elbo, ttheta_mean, ttheta_stddev) = sess.run([model.elbo, model.theta_q_mean, model.theta_q_stddev], feed_dict=feed_dict)
        theta_mean, _ = bf.transforms.logit(ttheta_mean)
        theta_z1, _ = bf.transforms.logit(ttheta_mean - 2*ttheta_stddev)
        theta_z3, _ = bf.transforms.logit(ttheta_mean + 2*ttheta_stddev)
        print "step %d elbo %.2f theta mean %.3f low %.3f high %.3f" % (i, elbo, theta_mean, theta_z1, theta_z3)

