import grpc
from concurrent import futures
import hydro_serving_grpc as hs
from tf_runtime_service import TFRuntimeService


class TensorflowRuntime:
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24

    def __init__(self, model_path):
        self.port = None
        self.server = None
        self.servicer = TFRuntimeService(model_path)

    def start(self, port="9090", max_workers=10):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        hs.add_PredictionServiceServicer_to_server(self.servicer, self.server)
        addr = "[::]:{}".format(port)
        print("Starting server on {}".format(addr))
        self.server.add_insecure_port(addr)
        self.server.start()

    def stop(self, code=0):
        self.server.stop(code)
