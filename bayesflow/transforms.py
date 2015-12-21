import numpy as np
import tensorflow as tf

def logit(x):

    if isinstance(x, tf.Tensor):
    
        transformed = 1.0 / (1 + tf.exp(-x))
        jacobian = transformed * (1-transformed)
        log_jacobian = tf.log(jacobian)
    else:
        transformed = 1.0 / (1 + np.exp(-x))
        jacobian = transformed * (1-transformed)
        log_jacobian = np.log(jacobian)

    return transformed, log_jacobian
