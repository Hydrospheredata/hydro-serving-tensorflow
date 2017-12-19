import tensorflow as tf
import numpy as np

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

