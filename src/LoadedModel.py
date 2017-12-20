import tensorflow as tf
import hydro_serving_grpc as hs


class Signature:
    def __init__(self, name, inputs: dict, outputs: dict):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self) -> str:
        return "Signature {}\n Inputs: {}\n Outputs: {}".format(self.name, self.inputs, self.outputs)


class LoadedModel:
    def __init__(self):
        self.model_path = None
        self.signatures = {}
        self.session = None
        self.contract = None

    def __str__(self) -> str:
        return "LoadedModel from {} with signatures: {}\n".format(self.model_path,
                                                                  list(map(lambda x: str(x), self.signatures)))

    @staticmethod
    def load(path: str):
        session = tf.Session()
        meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], path)
        signatures = {}
        for name, sig in meta_graph.signature_def.items():
            inputs = {}
            for inp_name, inp in sig.inputs.items():
                inputs[inp_name] = session.graph.get_tensor_by_name(inp.name)

            outputs = {}
            for out_name, out in sig.outputs.items():
                outputs[out_name] = session.graph.get_tensor_by_name(out.name)
            sig = Signature(name, inputs, outputs)
            signatures[name] = sig

        model = LoadedModel()
        model.model_path = path
        model.signatures = signatures
        model.session = session
        print("Loaded a model: {}".format(model))
        return model
