import numpy as np
import tensorflow as tf

def _infer_result_shape(*args):
    # given multiple Tensors and/or Python values, return the
    # shape to which an elementwise operation (e.g., addition)
    # would broadcast

    # TODO: check numpy API to see if there's a direct way to do this
    # otherwise I could create dummy tensors and pass them around?
    pass

def extract_shape(t):

    if t.get_shape():
        pass
    shape = tuple([d.value for d in t.get_shape()])
    if len(shape)==1 and shape[0] is None:
        shape = ()
    return shape

def logsumexp(x1, x2):
    shift = tf.maximum(x1, x2)
    return tf.log(tf.exp(x1 - shift) + tf.exp(x2-shift)) + shift
