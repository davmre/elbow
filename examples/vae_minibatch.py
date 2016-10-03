import numpy as np
import tensorflow as tf

from elbow import Model
from elbow.elementary import Gaussian
from elbow.joint_model import BatchGenerator
from elbow.models.neural import neural_bernoulli, neural_gaussian

from util import mnist_training_data

import time

def build_vae(d_z=2, d_hidden=256, d_x=784, N=100, total_N=60000):

    # MODEL
    z = Gaussian(mean=0, std=1.0, shape=(N,d_z), name="z", local=True)
    X = neural_bernoulli(z, d_hidden=d_hidden, d_out=d_x, name="X", local=True)

    # OBSERVED DATA
    x_placeholder = X.observe_placeholder()

    # VARIATIONAL MODEL
    q_z = neural_gaussian(X=x_placeholder, d_hidden=d_hidden, d_out=d_z, name="q_z")
    z.attach_q(q_z)

    jm = Model(X, minibatch_ratio = total_N/float(N))
    return jm, x_placeholder

def main():
    Xtrain, _, _, _ = mnist_training_data()

    batchsize = 100
    jm, x_batch = build_vae(N=batchsize, total_N=Xtrain.shape[0])

    batches = BatchGenerator(Xtrain, batch_size=batchsize)
    jm.register_feed(lambda : {x_batch: batches.next_batch()})

    jm.train(steps=10000, adam_rate=0.01)
    

if __name__ == "__main__":
    main()
