import os

from openvino.inference_engine import IECore


class Network:

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU"):
        """
        Loads the neural network model and initializes the Inference Engine plugin.
        :param model: the path to the XML file containing the model architecture
        :param device: the name of the device to run the inference on (default: CPU)
        """
        # load the model files
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # initialize the Inference Engine plugin and load the network
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)
        self.exec_network = self.plugin.load_network(network=self.network, device_name=device)

        # get the input and output blobs for the network
        self.input_blob = next(iter(self.network.input_info))
        self.output_blob = next(iter(self.network.outputs))

    def get_input_shape(self):
        """
        Returns the shape of the input for the loaded network
        :return: the shape of the input as a tuple
        """
        return self.network.input_info[self.input_blob].input_data.shape

    def inference(self, image):
        """
        Performs inference on a given input image using the loaded network
        :param image: the input image to perform inference on
        :return: the output produced by the network
        """
        obj_res = self.exec_network.infer({self.input_blob: image})
        return obj_res[self.output_blob]

    def wait(self):
        """
        Waits for the current inference request to complete
        :return: the status of the inference request
        """
        status = self.exec_network.requests[0].wait(-1)
        return status

    def extract_output(self):
        """
        Extracts the output from the current inference request
        :return: the output produced by the network
        """
        return self.exec_network.requests[0].output_blobs[self.output_blob]
