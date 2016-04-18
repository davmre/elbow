import numpy as np
import tensorflow as tf
import util

def logit(x, clip_finite=True):
    if isinstance(x, np.ndarray):
        transformed = 1.0 / (1 + np.exp(-x))
        jacobian = transformed * (1-transformed)
        log_jacobian = np.sum(np.log(jacobian))
    else:
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_logit_input")
        transformed = 1.0 / (1 + tf.exp(-x))
        jacobian = transformed * (1-transformed)
        if clip_finite:
            jacobian = tf.clip_by_value(jacobian, 1e-45, 1e38, name="clipped_jacobian")
        log_jacobian = tf.reduce_sum(tf.log(jacobian))
        

    return transformed, log_jacobian

def normalize(x_positive):
    if isinstance(x_positive, np.ndarray):
        n = x_positive.shape[0]
        Z = np.sum(x_positive)
        transformed = x_positive / Z    
        log_jacobian = -n * np.log(Z)
    else:
        n = util.extract_shape(x_positive)[0]
        Z = tf.reduce_sum(x_positive)
        transformed = x_positive / Z    
        log_jacobian = -n * tf.log(Z)
    return transformed, log_jacobian

def exp(x, clip_finite=True):
    if isinstance(x, np.ndarray):
        transformed = np.exp(x)
        log_jacobian = np.sum(x)
    else:
        transformed = tf.exp(x)
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_exp_input")
        log_jacobian = tf.reduce_sum(x)

    return transformed, log_jacobian


def square(x):
    transformed = x * x
    if isinstance(x, np.ndarray):
        log_jacobian = npsum(np.log(x)) + np.log(2)
    else:
        log_jacobian = tf.reduce_sum(tf.log(x)) + np.log(2)

    return transformed, log_jacobian

def sqrt(x):
    if isinstance(x, np.ndarray):
        transformed = np.sqrt(x)
        jacobian = .5 / transformed
        log_jacobian = np.sum(np.log(jacobian))
    else:
        transformed = tf.sqrt(x)
        jacobian = .5 / transformed
        log_jacobian = tf.reduce_sum(tf.log(jacobian))
    return transformed, log_jacobian

def reciprocal(x, clip_finite=True):
    
    if isinstance(x, np.ndarray):
        transformed = 1.0/x
        jacobian = 1.0 / (x*x)
        log_jacobian = np.sum(np.log(jacobian))
    else:
        if clip_finite:
            # caution: assumes input is positive
            x = tf.clip_by_value(x, 1e-38, 1e38, name="clipped_reciprocal_input")
            
        #transformed = 1.0/x
        nlogx = -tf.log(x)
        transformed = tf.exp(nlogx)
        # jacobian might under/overflow so just compute log directly
        log_jacobian = 2*tf.reduce_sum(nlogx)

    return transformed, log_jacobian

def chain_transforms(*args):

    def chain(x):
        log_jacobian = 0.0
        for transform in args:
            x, lj = transform(x)
            log_jacobian += lj
        return x, log_jacobian
    
    return chain

reciprocal_sqrt = chain_transforms(reciprocal, sqrt)
reciprocal_square = chain_transforms(reciprocal, square)
exp_reciprocal = chain_transforms(exp, reciprocal)
simplex = chain_transforms(exp, normalize)
