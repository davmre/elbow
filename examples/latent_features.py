import numpy as np
import tensorflow as tf

from elbow import Model, Gaussian, BernoulliMatrix, BetaMatrix
from elbow.models.factorizations import NoisyLatentFeatures

def latent_feature_model():
    K = 3
    D = 10
    N = 100

    a, b = np.float32(1.0), np.float32(1.0)

    pi = BetaMatrix(alpha=a, beta=b, shape=(K,), name="pi")
    B = BernoulliMatrix(p=pi, shape=(N, K), name="B")
    G = Gaussian(mean=0.0, std=1.0, shape=(K, D), name="G")
    D = NoisyLatentFeatures(B=B, G=G, std=0.1, name="D")
        
    jm = Model(D)

    return jm

def main():
    jm = latent_feature_model()

    sampled = jm.sample()
    jm["D"].observe(sampled["D"])
    
    sB = sampled["B"]
    sG = sampled["G"]
    sD = np.dot(sB, sG)
    mean_abs_err = np.mean(np.abs(sD - sampled["D"]))
    print "true latent values reconstruct observations with mean deviation %.3f" % (mean_abs_err)

    jm.train()
    posterior = jm.posterior()

    sess = jm.get_session()
    qB = sess.run(jm["B"].q_distribution().p)

    qG = posterior["q_G"]["mean"]
    qD = np.dot(qB, qG)
    mean_abs_err_inferred = np.mean(np.abs(qD - sampled["D"]))
    print "inferred latent values reconstruct observations with mean deviation %.3f" % (mean_abs_err_inferred)

    
if __name__ == "__main__":
    main()

