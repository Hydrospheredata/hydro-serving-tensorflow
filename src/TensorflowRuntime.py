import time
import grpc
from concurrent import futures
import hydro_serving_grpc as hs
from tf_runtime_service import TFRuntimeService


class TensorflowRuntime:
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24

    def __init__(self):
        self.port = None
        self.service = None

    def load_service(self, model_path):
        self.service = TFRuntimeService(model_path)

    def run(self, port="9090", max_workers=10):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        hs.add_PredictionServiceServicer_to_server(self.service, server)
        server.add_insecure_port("[::]:{}".format(port))
        server.start()
        try:
            while True:
                time.sleep(TensorflowRuntime._ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)
