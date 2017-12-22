import docker
import unittest
import tensorflow as tf
import time
import grpc
import hydro_serving_grpc as hs
import os


class DockerTests(unittest.TestCase):
    def test_docker_runtime(self):
        docker_client = docker.from_env()
        container = docker_client.containers.run(
            "hydrosphere/serving-grpc-runtime-tensorflow-latest-cpu:0.0.1",
            remove=True, detach=True,
            ports={'9090/tcp': 9090},
            volumes={os.path.abspath('models/tf_summator'): {'bind': '/model', 'mode': 'ro'}}
        )
        try:
            time.sleep(5)
            channel = grpc.insecure_channel('localhost:9090')
            client = hs.PredictionServiceStub(channel=channel)
            a = hs.TensorProto()
            a.ParseFromString(tf.contrib.util.make_tensor_proto(3, dtype=tf.int8).SerializeToString())
            b = hs.TensorProto()
            b.ParseFromString(tf.contrib.util.make_tensor_proto(2, dtype=tf.int8).SerializeToString())
            request = hs.PredictRequest(
                model_spec=hs.ModelSpec(signature_name="add"),
                inputs={
                    "a": a,
                    "b": b
                }
            )

            result = client.Predict(request)
            expected = hs.PredictResponse(
                outputs={
                    "sum": hs.TensorProto(
                        dtype=hs.DT_INT8,
                        tensor_shape=hs.TensorShapeProto(),
                        int_val=[5]
                    )
                }
            )
            self.assertEqual(result, expected)
        finally:
            print("Container logs:")
            print(container.logs().decode("utf-8"))
            container.stop()


if __name__ == "__main__":
    unittest.main()
