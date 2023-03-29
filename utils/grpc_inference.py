import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import gevent.ssl


def client_init(url="10.11.17.124:9101",
                ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False,
                verbose=False):
    """

    :param url:
    :param ssl: Enable encrypted link to the server using HTTPS
    :param key_file: File holding client private key
    :param cert_file: File holding client certificate
    :param ca_certs: File holding ca certificate
    :param insecure: Use no peer verification in SSL communications. Use with caution
    :param verbose: Enable verbose output
    :return:
    """
    if ssl:
        ssl_options = {}
        if key_file is not None:
            ssl_options['keyfile'] = key_file
        if cert_file is not None:
            ssl_options['certfile'] = cert_file
        if ca_certs is not None:
            ssl_options['ca_certs'] = ca_certs
        ssl_context_factory = None
        if insecure:
            ssl_context_factory = gevent.ssl._create_unverified_context
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=True,
            ssl_options=ssl_options,
            insecure=insecure,
            ssl_context_factory=ssl_context_factory)
    else:
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose)

    return triton_client



class LaneDetClient():
    def __init__(self):
        self.triton_client = client_init()
        self.inputs = []
        self.outputs = []
        self.inputs.append(grpcclient.InferInput('input', [1, 3, 360, 640], 'FP32'))
        self.outputs.append(grpcclient.InferRequestedOutput('predict_lanes'))
        self.outputs.append(grpcclient.InferRequestedOutput('914'))
        self.inputs[0].set_data_from_numpy(np.random.random([1, 3, 360, 640]).astype(np.float32))
        self.model_name = 'LaneDet'
    
    def __call__(self, input_tensor):
        self.inputs[0].set_data_from_numpy(input_tensor)
        result = self.triton_client.infer(self.model_name, inputs=self.inputs, outputs=self.outputs)
        output = result.as_numpy("predict_lanes")
        anchors = result.as_numpy("914")
        return output, anchors

if __name__ == "__main__":
    img_path = './model_transe/2022-04-22-11-00-05004(10).jpg'
    img = cv2.imread(img_path)
    client = LaneDetClient()
    output, anchors = client(img)