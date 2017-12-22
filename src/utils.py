import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import tensor_pb2

import numpy as np
import hydro_serving_grpc as hs


def convert_to_python(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.matrix):
        return data.tolist()
    else:
        print("{0} isn't convertible to python".format(type(data)))
        return data


def convert_data_to_tensor_shape(data, shape: tf.TensorShape):
    if shape.ndims == 2:
        return np.matrix(data)
    elif shape.ndims > 2:
        return np.array(data)
    else:
        return data


def load_and_optimize(model_path):
    with tf.Session() as temp_sess:
        meta_graph = tf.saved_model.loader.load(temp_sess, [tf.saved_model.tag_constants.SERVING], model_path)
        print("Model loaded.")
        signatures = {}
        for name, sig in meta_graph.signature_def.items():
            signatures["inputs"] = {}
            for inp_name, inp in sig.inputs.items():
                signatures["inputs"][inp_name] = temp_sess.graph.get_tensor_by_name(inp_name)

            signatures["outputs"] = {}
            for out_name, out in sig.outputs.items():
                signatures["outputs"][out_name] = temp_sess.graph.get_tensor_by_name(out_name)
        print("Loaded a model with signature: {}".format(signatures))


def make_tensor_proto(values, dtype=None, shape=None):
    if isinstance(values, hs.TensorProto):
        return values

    dtype = dtypes.as_dtype(dtype)

    if isinstance(values, np.ndarray):
        if dtype == tf.float32:
            return tensor_pb2.TensorProto(
                dtype=dtype.as_datatype_enum,
                tensor_shape=shape.as_proto(),
                float_val=values.flatten()
            )
        elif dtype == tf.double:
            return tensor_pb2.TensorProto(
                dtype=dtype.as_datatype_enum,
                tensor_shape=shape.as_proto(),
                double_val=values.flatten()
            )
    else:
        return tf.make_tensor_proto(values, dtype=dtype, shape=shape)
