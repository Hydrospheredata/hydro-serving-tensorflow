import os
from concurrent import futures
import time
import grpc

import hydro_serving_grpc as hs
from tf_runtime_service import TFRuntimeService
from utils import *

PORT = int(os.getenv("APP_GRPC_PORT", "9090"))
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

print("Importing TensorFlow model...")
sess, inputs, outputs = load_and_optimize("/model")

if __name__ == '__main__':
    print("Runtime is ready to serve...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hs.add_PredictionServiceServicer_to_server(TFRuntimeService("/model", "/contract/contract.protobin"), server)
    server.add_insecure_port(str(PORT))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
