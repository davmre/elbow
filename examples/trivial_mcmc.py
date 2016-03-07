import numpy as np
import tensorflow as tf
import bayesflow as bf
from bayesflow.hmc import HMCInferenceFunctional
from bayesflow.mh import RandomWalkMHInference

"""
Demonstration of using MH to approximate a multimodal posterior.
"""

def toy_bimodal_lik(x1, x2):
    x1_mode1 = bf.dists.gaussian_log_density(x1, mean=2.0, stddev=1.0)
    x1_mode2 = bf.dists.gaussian_log_density(x1, mean=-2.0, stddev=1.0)
    x1_ll = tf.log(tf.exp(x1_mode1) + tf.exp(x1_mode2))

    x2_mode1 = bf.dists.gaussian_log_density(x2, mean=2.0, stddev=1.0)
    x2_mode2 = bf.dists.gaussian_log_density(x2, mean=-2.0, stddev=1.0)
    x2_ll = tf.log(tf.exp(x2_mode1) + tf.exp(x2_mode2))

    return x1_ll + x2_ll

def infer_mh():
    mh = RandomWalkMHInference(toy_bimodal_lik)
    mh.add_latent("x1", 0.0)
    mh.add_latent("x2", 0.0)
    mh.build_mh_update()
    mh.train(steps=2000)
    return mh
    
def infer_hmc():
    hmc = HMCInferenceFunctional(toy_bimodal_lik)
    hmc.add_latent("x1", 0.0)
    hmc.add_latent("x2", 0.0)
    hmc.build_hmc_update(L=2, eps=1.5)
    hmc.train(steps=2000)
    return hmc

def main():
    print "running mh inference"
    infer_mh()

    print "running hmc inference"
    infer_hmc()
    
if __name__ == "__main__":
    main()
