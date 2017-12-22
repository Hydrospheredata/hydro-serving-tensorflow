import hydro_serving_grpc as hs
from LoadedModel import LoadedModel
import tensorflow as tf
import utils
import grpc


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
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("{} signature is not present in the model".format(signature_name))
            return hs.PredictResponse()
        print("Using {} signature".format(sig.name))
        feed = {}
        for (k, v) in sig.inputs.items():
            tensor = request.inputs[k]
            feed[v.name] = tf.contrib.util.make_ndarray(tensor)

        result = self.model.session.run(list(sig.outputs.values()), feed_dict=feed)

        converted_results = {}
        for k, v in zip(sig.outputs.keys(), result):
            original_tensor = utils.make_tensor_proto(v, dtype=sig.outputs[k].dtype, shape=sig.outputs[k].shape)
            tensor_proto = hs.TensorProto()
            tensor_proto.ParseFromString(original_tensor.SerializeToString())
            print("Answer: {}".format(tensor_proto))
            converted_results[k] = tensor_proto

        return hs.PredictResponse(outputs=converted_results)
