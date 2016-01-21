import numpy as np
import tensorflow as tf
import bayesflow as bf
from bayesflow.mh import RandomWalkMHInference

"""
Demonstration of using MH to approximate a multimodal posterior.
"""

def toy_bimodal_lik(x1, x2):
    x1_mode1 = bf.dists.gaussian_log_density(x1, mean=1.0, stddev=2.0)
    x1_mode2 = bf.dists.gaussian_log_density(x1, mean=-1.0, stddev=2.0)

    x2_mode1 = bf.dists.gaussian_log_density(x2, mean=1.0, stddev=2.0)
    x2_mode2 = bf.dists.gaussian_log_density(x2, mean=-1.0, stddev=2.0)

    return x1_mode1 + x1_mode2 + x2_mode1 + x2_mode2

def main():
    mh = RandomWalkMHInference(toy_bimodal_lik)
    mh.add_latent("x1", 0.0)
    mh.add_latent("x2", 0.0)
    mh.build_mh_update()
    mh.train(steps=5000)

    
if __name__ == "__main__":
    main()
