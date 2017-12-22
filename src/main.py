import os
from TensorflowRuntime import TensorflowRuntime

PORT = os.getenv("APP_GRPC_PORT", "1123")

if __name__ == '__main__':
    runtime = TensorflowRuntime()
    print("Reading the model...")
    runtime.load_service("/model")
    print("Runtime is ready to serve...")
    runtime.run(port=PORT)
