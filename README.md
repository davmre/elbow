# Bayesflow

Bayesflow is a minimalist framework for probabilistic programming, built atop TensorFlow. The goal is to make it easy to define and compose sophisticated probability models and inference strategies. 

The current status extremely immature, pre-pre-pre-alpha, many components are not yet implemented or even designed and major breaking changes may happen at any time. If you want to help out and/or and have ideas for what to build, please get in touch!

## Motivation

The computational graph abstraction implemented by TensorFlow, Theano, and others has been an powerful catalyst for research in deep learning, removing the need for tedious and error-prone gradient calculations and allowing for reuse of modular high-level components. 

Similar ideas have recently risen to prominence in probabilistic programming; for example, [Stan](http://mc-stan.org/) uses automatic differentiation to perform practical inference in a large class of  probability models. 

This project aims to provide abstractions for probabilistic programming that inherit the modularity and flexibility that have made computational graphs so successful in deep learning research. The initial target audience are researchers who want to move beyond the modeling restrictions and one-size-fits-all inference strategies available in existing systems. Building atop TensorFlow also opens up the prospect of easily deploying Bayesian models to end users. 

Goals include:
- Expose simple abstractions for defining probability models and custom inference strategies.
- Support for expressive generative models including black-box "neural" components, discrete variables, Bayesian nonparametrics, etc.
- Provide a testbed for inference strategies including mean-field and structured variational approximations, [score function estimators](http://approximateinference.org/accepted/WeberEtAl2015.pdf), [hierarchical variational models](http://arxiv.org/abs/1511.02386), [inference networks](http://arxiv.org/abs/1312.6114), [alternative divergences](http://arxiv.org/abs/1511.03243), [stochastic dynamics](http://www.icml-2011.org/papers/398_icmlpaper.pdf), etc., separating inference code from model definitions.
- Straightforward integration with the Python machine learning ecosystem: [scikit-learn](http://scikit-learn.org/stable/), [GPy](https://sheffieldml.github.io/GPy/), etc.
- Efficient and scalable inference using GPUs and/or distributed clusters. 

## Status

Currently implemented:
- *bayesflow.mean_field*: mean field Gaussian posterior inference via the reparameterization trick
- *bayesflow.dists*: common probability densities as building blocks for differentiable likelihoods
- *baysesflow.transforms*: differentiable transformations for constrained latent variables
- *examples*: implementations of several models from Stan and elsewhere, both in raw Tensorflow and using the Bayesflow abstractions


