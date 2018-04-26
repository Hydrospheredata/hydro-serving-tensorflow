import hydro_serving_grpc as hs

from LoadedModel import LoadedModel
import tensorflow as tf
import utils
import grpc
import logging


class TFRuntimeService(hs.PredictionServiceServicer):
    def __init__(self, model_path):
        self.model_path = "{}/files".format(model_path)
        self.logger = logging.getLogger("TensorflowRuntime")
        self.session = tf.Session()
        self.model = LoadedModel.load(self.session, self.model_path)

        if self.model.is_stateful():
            self.state = {x.name: self.session.run(x) for x in self.model.zero_states}
            self.state_fetch = {x.name: x for x in self.model.state_placeholders}

    def Predict(self, request, context):
        self.logger.info("Received inference request: {}".format(request))
        signature_name = request.model_spec.signature_name
        if signature_name in self.model.signatures:
            sig = self.model.signatures[signature_name]
        else:
            msg = "{} signature is not present in the model".format(signature_name)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(msg)
            self.logger.error(msg)
            return hs.PredictResponse()
        self.logger.debug("Using {} signature".format(sig.name))
        fetch = sig.outputs

        feed = {}
        for (k, v) in sig.inputs.items():
            tensor = request.inputs[k]
            feed[v.name] = tf.contrib.util.make_ndarray(tensor)

        if self.model.is_stateful():
            fetch = {**fetch, **self.state_fetch}
            feed.update(self.state)

        result = self.model.session.run(fetch, feed_dict=feed)

        converted_results = {}
        for out_key, out_tensor in sig.outputs.items():
            out_value = result[out_key]
            original_tensor = utils.make_tensor_proto(out_value, dtype=out_tensor.dtype, shape=out_tensor.shape)
            tensor_proto = hs.TensorProto()
            tensor_proto.ParseFromString(original_tensor.SerializeToString())
            self.logger.info("Answer: {}".format(tensor_proto))
            converted_results[out_key] = tensor_proto

        for i, v in enumerate(self.model.state_placeholders):
            state_name = self.model.zero_states[i]
            self.state[state_name.name] = result[v.name]

        return hs.PredictResponse(outputs=converted_results)
