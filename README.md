# Elbow

Elbow is a flexible framework for probabilistic programming, built atop TensorFlow. The goal is to make it easy to define and compose sophisticated probability models and inference strategies. Currently the focus is on variational inference, in which we explicitly optimize an evidence bound (the ELBO) corresponding to the KL divergence between an approximate posterior representation and the true (unknown) posterior. Elbow supports the construction of sophisticated approximate posteriors, including [inference networks](https://arxiv.org/abs/1312.6114) and [structured message passing](https://arxiv.org/abs/1603.06277) as well as general-purpose [variational models](https://arxiv.org/abs/1511.02386) using the same model components as base-level models. 

# Usage

Elbow is oriented around directed graphical models. Currently the focus is on models with continuous-valued variables. The fundamental unit is the `ConditionalDistribution`, representing a random variable conditioned on a set of inputs (which may be the empty set). For example, a `Gaussian` distribution takes as inputs its mean and standard deviation. To model a Gaussian with unknown mean, we write

```python
from elbow import Gaussian, Model

mu = Gaussian(mean=0, std=10, name="mu")
X = Gaussian(mean=mu, std=1, shape=(100,), name="X")
```

Inputs can be other random variables, Python literals, Tensors or numpy arrays. Note that we specify the shape of X in order to draw multiple samples. 

A set of conditional distributions defines a joint distribution. We represent this using a `Model`, which provides convenience methods for sampling from and performing inference over a directed probability model. First we can sample some data from our model:

```python
m = Model(X) # automatically includes all ancestors of X
sampled = m.sample()

print "mu is", sampled["mu"]
print "empirical mean is", np.mean(sampled["X"])
```

Now let's try running inference given the sampled data. To do this, we first specify a form for the variational posterior. We do this using the same `ConditionalDistribution` components as the object-level model. In this example, we'll let the posterior on X be a point mass at the observed values, and the posterior on mu be Gaussian. 

```python
X.observe(sampled["X"])
mu.attach_q(Gaussian(shape=mu.shape, name="q_mu")) # not actually necessary, would be inferred automatically, but shown here for illustration
```

We then call the model's `train` method, which automatically constructs and optimizes a variational bound inside a Tensorflow session:

```python
m.train(steps=500)
posterior = m.posterior()
print "posterior on mu has mean %.2f and std %.2f" % (posterior["q_mu"]["mean"], posterior["q_mu"]["std"])
```

Since the conjugate prior for a Gaussian mean is itself Gaussian, in this simple example the variational posterior should be exact. 

# Documentation

TODO write more documentation including examples of inference networks, transformed variables, and minibatch training.

Currently the best reference is the source code itself. The `examples` directory contains implementations of models including sparse matrix factorization, clustering (Gaussian mixture model), binary latent features, and variational autoencoders, among others. 

# Custom components

Elbow makes it easy to define new types of random variables. A new variable must extend the `ConditionalDistribution` class, specify its inputs, and provide methods for sampling and computing the log probability density. For example, let's implement a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution), which is like a Gaussian but with longer tails.

```python
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
```

Each input is provided a default parameterization, to be used when optimizing over an unspecified value, as in a variational model. Here location is unconstrained, but scale must be positive, so we parameterize it as the exponential of an unconstrained Tensor. Other possibilities include `unit_interval`, `simplex_constrained`, and `psd_matrix`. 

Note that the sampling method generates samples by applying a transformation (in this case the inverse CDF) to a random source (uniform in this case), so that the sample is differentiable with respect to the location and scale parameters. This is the so-called "reparameterization trick", which allows us to optimize the parameters with respect to a a Monte Carlo approximation of the evidence lower bound (ELBO). Currently we assume all sampling methods are reparameterized in this way, though we plan to implement alternative gradient estimators (e.g., REINFORCE/BBVI) in the future.

To use our new distribution in variational models, we can also optionally implement an analytic entropy (if we did not do this, Elbow would default to the Monte Carlo entropy given by evaluating the log density at a sampled value). As a convenience we also hint that the default variational model for a Laplace-distributed variable should itself be a Laplace distribution.

```python
   def _entropy(self, loc, scale):
       return 1 + tf.reduce_sum(tf.log(2*scale))

   def default_q(self, **kwargs):
       return Laplace(shape=self.shape, name="q_"+self.name)
```

We can now use our new variable type and compose it with other distributions. For example, we can redo the example from above, estimating the mean of a Gaussian, but now with a Laplace prior.

```python
mu = Laplace(loc=0, scale=10, name="mu")
X = Gaussian(mean=mu, std=1, shape=(100,), name="X")

m = Model(X)
sampled = m.sample()
X.observe(sampled[X])

m.train(steps=500)
posterior = m.posterior()
print "sampled mu was %.2f; posterior has loc %.2f and scale %.2f" % (sampled["mu"], posterior["q_mu"]["scale"], posterior["q_mu"]["scale"])
```







