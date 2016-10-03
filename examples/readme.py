import tensorflow as tf
import numpy as np

from elbow import Gaussian, Model

"""
Runnable version of the examples from the README file.
"""

def gaussian_mean_inference():
    mu = Gaussian(mean=0, std=10, name="mu")
    X = Gaussian(mean=mu, std=1, shape=(100,), name="X")

    m = Model(X)
    sampled = m.sample()

    print "mu is", sampled["mu"]
    print "empirical mean is", np.mean(sampled["X"])

    X.observe(sampled["X"])
    mu.attach_q(Gaussian(shape=mu.shape, name="q_mu"))

    m.train()
    posterior = m.posterior()
    print "posterior on mu has mean %.2f and std %.2f" % (posterior["q_mu"]["mean"], posterior["q_mu"]["std"])

def custom_laplace():
    from elbow import ConditionalDistribution
    from elbow.parameterization import unconstrained, positive_exp
    
    class Laplace(ConditionalDistribution):

        def inputs(self):
            return {"loc": unconstrained, "scale": positive_exp}

        def _sample(self, loc, scale):
            base = tf.random_uniform(shape=self.shape, dtype=tf.float32) - 0.5
            std_laplace = tf.sign(base) * tf.log(1-2*tf.abs(base))
            return loc + scale * std_laplace

        def _logp(self, result, loc, scale):
            return -tf.reduce_sum(tf.abs(result-loc)/scale - tf.log(2*scale))

        def _entropy(self, loc, scale):
            return 1 + tf.reduce_sum(tf.log(2*scale))

        def default_q(self, **kwargs):
            return Laplace(shape=self.shape, name="q_"+self.name)

    mu = Laplace(loc=0, scale=10, name="mu")
    X = Gaussian(mean=mu, std=1, shape=(100,), name="X")

    m = Model(X)
    sampled = m.sample()
    X.observe(sampled["X"])

    m.train(steps=500)
    posterior = m.posterior()
    print "sampled mu was %.2f; posterior has loc %.2f and scale %.2f" % (sampled["mu"], posterior["q_mu"]["loc"], posterior["q_mu"]["scale"])


gaussian_mean_inference()
custom_laplace()

