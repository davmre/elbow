# Bayesflow

Bayesflow is a minimalist framework for probabilistic programming, built atop TensorFlow. The goal is to make it easy to define and compose sophisticated probability models and inference strategies. 

Current status is extremely immature, pre-pre-pre-alpha, many components are not yet implemented or even designed and major breaking changes may happen at any time. If you want to help out and/or and have ideas for what to build, please get in touch!

## Motivation

The computational graph abstraction implemented by TensorFlow, Theano, and others has been an powerful catalyst for research in deep learning, removing the need for tedious and error-prone gradient calculations and allowing for reuse of modular high-level components. 

Similar ideas have recently risen to prominence in probabilistic programming; for example, [Stan](http://mc-stan.org/) uses automatic differentiation to construct Hamiltonian MCMC or black-box variational inference algorithms for a wide range of models. However, research work in probabilistic modeling and inference often involves experimenting with novel model components and/or inference strategies not available in a prebuilt package. 

Bayesflow provides reusable building blocks for representing probability models and inference algorithms within a TensorFlow graph. The underlying graph is fully exposed to the user and can be arbitrarily modified and inspected. In particular, it is easy to integrate traditional modeling components with black-box "neural" decoders and/or encoding networks. The TensorFlow execution layer also enables (at least in principle) deployment to GPUs, clusters, and mobile devices.

## Components
Currently implemented (at least partially):
- Building blocks for expressing differentiable likelihoods:
  - `bf.dists`: Common probability densities
  - `bf.transforms`: differentiable transformations for constrained latent variables
- Inference algorithms:
  - `bayesflow.mean_field`: mean field Gaussian posterior inference via the reparameterization trick

The `examples` directory contains several models (linear regression, matrix factorization, beta-bernoulli, variational autoencoders) implemented using Bayesflow components to varying extents.




