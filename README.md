# Bayesflow

Modular Bayesian inference using TensorFlow

## Motivation

The computational graph abstraction implemented by TensorFlow, Theano, and others has been an powerful catalyst for research in deep learning, removing the need for tedious and error-prone gradient calculations, and allowing for reuse of modular high-level components. 

Concurrently, automatic differentiation has also risen to prominence in the probabilistic programming community; for example, [Stan](http://mc-stan.org/) generates practical inference algorithms for a large class of differentiable probability models. 

This project aims to provide abstractions for probabilistic programming that inherit the modularity and flexibility that have made computational graphs so successful in deep learning research. The target audience is researchers with the knowledge and desire to design and experiment with custom inference strategies. Goals include:
- Provide simple abstractions for defining probability models and performing inference.
- Support for expressive generative models including black-box "neural" components, discrete variables, Bayesian nonparametrics, etc.
- A testbed for custom inference strategies including structured variational approximations, [score function estimators](http://approximateinference.org/accepted/WeberEtAl2015.pdf), [hierarchical variational models](http://arxiv.org/abs/1511.02386), [inference networks](http://arxiv.org/abs/1312.6114), [alternative divergences](http://arxiv.org/abs/1511.03243), [stochastic dynamics](http://www.icml-2011.org/papers/398_icmlpaper.pdf), etc.
- Straightforward integration with the Python machine learning ecosystem: [scikit-learn](http://scikit-learn.org/stable/), [GPy](https://sheffieldml.github.io/GPy/), etc.
- Efficient and scalable inference using GPUs and/or distributed clusters. 

## Status

Currently implemented:
- *bayesflow.dists*: common probability densities as building blocks for differentiable likelihoods
- *baysesflow.transforms*: differentiable transformations for constrained latent variables
- *bayesflow.mean_field*: mean field Gaussian posterior inference via the reparameterization trick
- *examples*: implementations of several models from Stan and elsewhere, both in raw TensorFLow and using the Bayesflow abstractions

Everything is extremely immature, pre-pre-pre-alpha, many components are not yet implemented or even designed and major breaking changes may happen at any time. If you want to help out and/or and have ideas for what to build, please get in touch!
