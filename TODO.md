


- Higher-level model specification interface that eliminates the need
  to write separate likelihood and sampling functions

- ability to specify per-variable approximating distributions
  (technically possible now by composing inverse Gaussian CDF with
  desired CDF, but that's obviously silly)

- Support for expressive generative models:
  - black-box "neural" components (already implicitly available, but build an example?)
  - discrete variables
  - Bayesian nonparametrics

- Other inference techniques:
  - stochastic gradient
  - HMC/NUTS
  - Gibbs sampling for discrete variables within the TF graph?
  - score function estimators and structured variational approximations [inspired by RL](http://approximateinference.org/accepted/WeberEtAl2015.pdf), 
  - [hierarchical variational models](http://arxiv.org/abs/1511.02386), 
  - [inference networks](http://arxiv.org/abs/1312.6114), 
  - [alternative divergences](http://arxiv.org/abs/1511.03243), 
  - [stochastic dynamics](http://www.icml-2011.org/papers/398_icmlpaper.pdf)

- Straightforward integration with the Python machine learning ecosystem: [scikit-learn](http://scikit-learn.org/stable/), [GPy](https://sheffieldml.github.io/GPy/), etc.

- Efficient and scalable inference using GPUs and/or distributed clusters. 
