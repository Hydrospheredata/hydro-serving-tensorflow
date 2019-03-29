import os
from TensorflowRuntime import TensorflowRuntime
import time
import logging

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
PORT = int(os.getenv("APP_PORT", "9090"))

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logger = logging.getLogger("main")
    logger.info("Reading the model...")
    runtime = TensorflowRuntime("/model", port=PORT)
    logger.info("Runtime is ready to serve...")
    runtime.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        runtime.stop()
