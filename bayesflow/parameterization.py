import numpy as np
import tensorflow as tf

from util import concrete_shape

"""
Utility methods for defining constrained variables as transforms of an unconstrained parameterization. 
"""

def unconstrained(shape=None, init=None, name=None):

    if init is None:
        shape = concrete_shape(shape)
        init = np.float32(np.random.randn(*shape))
        
    val = tf.Variable(init, name=name)
    return val
    
def positive_exp(shape=None, init_log=None, name=None):
    # a Tensor of values that are pointwise positive, represented by an exponential

    
    if init_log is None:
        shape = concrete_shape(shape)
        init_log = np.float32(np.ones(shape) * -10)
    
    log_value = tf.Variable(init_log, name= "log_"+name if name is not None else None)
    pos_value = tf.exp(tf.clip_by_value(log_value, -42, 42), name=name)
    return pos_value

