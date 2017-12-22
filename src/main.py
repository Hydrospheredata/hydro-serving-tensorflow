import os
from TensorflowRuntime import TensorflowRuntime
import time

PORT = os.getenv("APP_GRPC_PORT", "9090")

if __name__ == '__main__':
    print("Reading the model...")
    runtime = TensorflowRuntime("/model")
    print("Runtime is ready to serve...")
    runtime.start(port=PORT)
    try:
        while True:
            time.sleep(TensorflowRuntime._ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        runtime.stop()
