import numpy as np
import tensorflow as tf

def _infer_result_shape(*args):
    # given multiple Tensors and/or Python values, return the
    # shape to which an elementwise operation (e.g., addition)
    # would broadcast

    # TODO: check numpy API to see if there's a direct way to do this
    # otherwise I could create dummy tensors and pass them around?
    pass

def _tf_extract_shape(t):

    if t.get_shape():
        pass
    shape = [d.value for d in t.get_shape()]
    if len(shape)==1 and shape[0] is None:
        shape = []
    return shape

