import tensorflow as tf
import numpy as np

import bayesflow.dists as bfd

import util

class LinearRegressionARD(object):

    def __init__(self, N, d, a0=1.0, b0=1.0, c0=1.0, d0=1.0, batch_size=64):

        self.a0, self.b0, self.c0, self.d0 = a0, b0, c0, d0
        self.N, self.d, self.batch_size = N, d, batch_size

        # q distribution for alpha
        alpha_default = a0/b0
        transformed_alpha_default = (np.ones((d,)) * 1.0/np.sqrt(alpha_default)).astype(np.float32)
        self.one_over_sqrt_alpha_q_mean = tf.Variable(transformed_alpha_default, "one_over_sqrt_alpha_q_mean")
        self.one_over_sqrt_alpha_q_stddev = tf.Variable((np.ones((d,)) * 1e-3).astype(np.float32), "one_over_sqrt_alpha_q_stddev")
        self.one_over_sqrt_alpha_q_entropy = bfd.gaussian_entropy(stddev=self.one_over_sqrt_alpha_q_stddev)
        self.alpha_eps = tf.placeholder(dtype=tf.float32, shape=(d,), name="alpha_eps")
        
        # model for alpha
        self.one_over_sqrt_alpha = self.alpha_eps * self.one_over_sqrt_alpha_q_stddev + self.one_over_sqrt_alpha_q_mean
        self.alpha_prior = bfd.gamma_log_density(1.0/tf.square(self.one_over_sqrt_alpha), alpha=a0, beta=b0)
        # TODO ADD JACOBIAN
        
        # q distribution for sigma
        sigma2_default = np.array(d0/(c0+1)).astype(np.float32)
        self.sigma_q_mean = tf.Variable(np.sqrt(sigma2_default),
                                        name="sigma_q_mean")
        self.sigma_q_stddev = tf.Variable(np.array(1e-3).astype(np.float32),
                                          name="sigma_q_stddev")
        self.sigma_q_entropy = bfd.gaussian_entropy(stddev=self.sigma_q_stddev)
        self.sigma_eps = tf.placeholder(dtype=tf.float32, name="sigma_eps")
        # TODO ADD JACOBIAN
        
        # model for sigma2
        self.sigma = self.sigma_eps * self.sigma_q_stddev + self.sigma_q_mean
        self.sigma2_prior = bfd.inv_gamma_log_density(tf.square(self.sigma), alpha=c0, beta=d0)

        # q distribution for w
        self.w_q_mean = tf.Variable(tf.random_normal([d,], stddev=1.0, dtype=tf.float32), "w_q_mean")
        self.w_q_stddev = tf.Variable(np.ones((d,), dtype=np.float32) * 1e-3, "w_q_stddev")
        self.w_q_entropy = bfd.gaussian_entropy(stddev=self.w_q_stddev)
        self.w_eps = tf.placeholder(dtype=tf.float32, shape=(d,), name="w_eps")
        
        self.w = self.w_eps * self.w_q_stddev + self.w_q_mean
        self.w_lik = bfd.gaussian_log_density(self.w, mean=0, stddev=self.sigma * self.one_over_sqrt_alpha)
        
        self.batch_X = tf.placeholder(tf.float32, (batch_size, d), name="X")
        self.batch_y = tf.placeholder(tf.float32, (batch_size, ), name="y")

        ybar = tf.matmul(self.batch_X,tf.expand_dims(self.w, 1))
        self.y_lik = bfd.gaussian_log_density(self.batch_y,
                                              mean=ybar,
                                              stddev=self.sigma)

        self.entropy = self.w_q_entropy + self.sigma_q_entropy + self.one_over_sqrt_alpha_q_entropy
        self.global_lik = self.alpha_prior + self.sigma2_prior + self.w_lik

        self.stochastic_elbo = self.y_lik + float(batch_size)/N * (self.global_lik + self.entropy)

    def simulate(self, X, alpha, sigma2):
        w_stddev = np.sqrt(sigma2) * 1.0/np.sqrt(alpha)
        w = np.random.randn(d) * w_stddev
        f = np.dot(X, w)
        y = f + np.random.randn() * np.sqrt(sigma2)

        return w, y

N = 10000
d = 250
alpha = np.ones((d),)
alpha[d/2:] = 10.0
sigma2 = 1.0

X = np.random.rand(N, d)

model = LinearRegressionARD(N, d)
w, y = model.simulate(X, alpha, sigma2)

train_step = tf.train.AdamOptimizer(1e-1).minimize(-model.stochastic_elbo)
init = tf.initialize_all_variables()
        
sess = tf.Session()
sess.run(init)


for i, batch_xs, batch_ys in util.batch_generator(X, y, 64, max_steps=20000):
    feed_dict = {model.batch_X: batch_xs,
                 model.batch_y: batch_ys,
                 model.sigma_eps: np.random.randn(),
                 model.alpha_eps: np.random.randn(model.d),
                 model.w_eps: np.random.randn(model.d)}
    sess.run(train_step, feed_dict = feed_dict)
        
    if i % 100 == 0:

        (elbo, one_over_sqrt_alpha_q_mean, sigma_mean, w_mean) = sess.run([model.stochastic_elbo, model.one_over_sqrt_alpha_q_mean, model.sigma_q_mean, model.w_q_mean], feed_dict=feed_dict)
        alpha_err_norm = np.linalg.norm(1.0/one_over_sqrt_alpha_q_mean**2 - alpha)
        w_err_norm = np.linalg.norm(w_mean - w)
        print "step %d elbo %.2f alpha_err_norm %.2f w_err_norm %.2f" % (i, elbo, alpha_err_norm, w_err_norm)

    if i % 1000 == 999:
        import pdb; pdb.set_trace()
