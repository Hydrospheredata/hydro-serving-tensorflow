import tensorflow as tf


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
        self.state_placeholders = []
        self.zero_states = []
        self.contract = None

    def __str__(self) -> str:
        signatures = list(map(lambda x: str(x), self.signatures))
        if self.is_stateful():
            return "Stateful LoadedModel from {} with signatures: {}\n".format(self.model_path, signatures)
        else:
            return "Stateless LoadedModel from {} with signatures: {}\n".format(self.model_path, signatures)

    def is_stateful(self):
        return len(self.state_placeholders) > 0

    @staticmethod
    def load(session, path: str):
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
        model.state_placeholders = session.graph.get_collection("h_state_placeholders")  # FIXME move constant to a lib
        model.zero_states = session.graph.get_collection("h_zero_states")  # FIXME move constant to a lib
        print("Loaded a model: {}".format(model))
        return model
