import numpy as np
import tensorflow as tf

import bayesflow as bf

from bayesflow.models.joint_model import JointModel
from bayesflow.models.elementary import Gaussian
from bayesflow.models import JMContext, current_scope

def model_extend(N):
    mu = Gaussian(mean=0.0, std=10.0, shape=(1,), name='mu')
    x = Gaussian(mean=mu, std=1.0, shape=(N,), name='x')

    model = JointModel()
    model.extend(mu)
    model.extend(x)

    q_mu = Gaussian(shape=mu.shape, name='q_mu')
    model.marginalize(mu, q_mu)

    return model

def model_implicit(N):

    with JMContext() as jm:
        mu = Gaussian(mean=0.0, std=10.0, shape=(1,), name='mu')
        x = Gaussian(mean=mu, std=1.0, shape=(N,), name='x')

    jm.marginalize(mu)
    return jm

def model_compose(N):
    mu = Gaussian(mean=0.0, std=10.0, shape=(1,))
    q_mu = Gaussian(shape=mu.shape)
    
    x = Gaussian(mean=mu, std=1.0, shape=(N,))

    model = JointModel()
    model.extend(mu)
    newmodel = model.compose(x, {mu: q_mu})

    return newmodel
    
def main():
    N = 100

    model = model_implicit(N=N)

    sX = np.float32(np.random.randn(N) + 4.3)

    outputs = model.outputs()
    x = outputs[0]
    model.observe(x, sX)

    print model.outputs()

    
    posterior = model.train()

    #print posterior
    
    #print model.posterior('mu')
    
if __name__=="__main__":
    main()
