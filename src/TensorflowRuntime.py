import grpc
from concurrent import futures
import hydro_serving_grpc as hs
from tf_runtime_service import TFRuntimeService
import logging


class TensorflowRuntime:
    def __init__(self, model_path, port=9090, max_workers=10):
        self.servicer = TFRuntimeService(model_path)
        self.logger = logging.getLogger("TensorflowRuntime")
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        hs.add_PredictionServiceServicer_to_server(self.servicer, self.server)
        addr = "[::]:{}".format(port)
        self.logger.info("Starting server on {}".format(addr))
        self.port = self.server.add_insecure_port(addr)

    def start(self):
        return self.server.start()

    def stop(self, code=0):
        self.server.stop(code)
