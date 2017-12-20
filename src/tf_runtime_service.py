import hydro_serving_grpc as hs

from LoadedModel import LoadedModel

import tensorflow as tf


class TFRuntimeService(hs.PredictionServiceServicer):
    def __init__(self, model_path):
        self.model_path = model_path

        self.model = LoadedModel.load(model_path)

    def Predict(self, request, context):
        print("Received inference request: {}".format(request))
        signature_name = request.model_spec.signature_name
        if signature_name in self.model.signatures:
            sig = self.model.signatures[signature_name]
        else:
            print("Requested entry point ({}) is not in the graph. Using first available.".format(signature_name))
            sig = list(self.model.signatures.values())[0]
        print("Using {} signature".format(sig.name))
        feed = {}
        for (k, v) in sig.inputs.items():
            tensor = request.inputs[k]
            feed[v.name] = tf.contrib.util.make_ndarray(tensor)

        result = self.model.session.run(list(sig.outputs.values()), feed_dict=feed)

        converted_results = {}
        for k, v in zip(sig.outputs.keys(), result):
            tensor_proto = hs.TensorProto()
            tensor_proto.ParseFromString(tf.contrib.util.make_tensor_proto(v).SerializeToString())
            converted_results[k] = tensor_proto

        return hs.PredictResponse(outputs=converted_results)
