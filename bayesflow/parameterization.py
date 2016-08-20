import numpy as np
import tensorflow as tf

from util import concrete_shape
from transforms import simplex, logit
"""
Utility methods for defining constrained variables as transforms of an unconstrained parameterization. 
"""

def unconstrained(shape=None, init=None, name=None):

    if init is None:
        shape = concrete_shape(shape)
        init = np.float32(np.random.randn(*shape))
        
    val = tf.Variable(init, name=name)
    return val

def simplex_constrained(shape=None, init_log=None, name=None):

    if init_log is None:
        shape = concrete_shape(shape)
        init_log = np.float32(np.ones(shape) * -10)

    log_value = tf.Variable(init_log, name= "log_"+name if name is not None else None)
    simplexed, log_jacobian = simplex(log_value)
    return simplexed

def unit_interval(shape=None, init_log=None, name=None):
    # Defines a matrix each element of which is in the unit interval.
    # This is different from simplex_constrained which defines a
    # vector guaranteed to be in the unit simplex.

    if init_log is None:
        shape = concrete_shape(shape)
        init_log = np.float32(np.ones(shape) * -10)

    log_value = tf.Variable(init_log, name= "log_"+name if name is not None else None)
    v, log_jacobian = logit(log_value)
    return v
        
def positive_exp(shape=None, init_log=None, name=None):
    # a Tensor of values that are pointwise positive, represented by an exponential

    
    if init_log is None:
        shape = concrete_shape(shape)
        init_log = np.float32(np.ones(shape) * -10)
    
    log_value = tf.Variable(init_log, name= "log_"+name if name is not None else None)
    pos_value = tf.exp(tf.clip_by_value(log_value, -42, 42), name=name)
    return pos_value

