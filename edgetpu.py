import tflite_runtime.interpreter as tflite
import platform
import numpy as np

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


class EdgeTpuModel:
    def __init__(self, model_file):
        self.make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.name = "EdgeTPU"

    def make_interpreter(self, model_file):
        model_file, *device = model_file.split("@")
        self.interpreter = tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(
                    EDGETPU_SHARED_LIB, {"device": device[0]} if device else {}
                )
            ],
        )

    def predict(self, input_board):
        input_details = self.interpreter.get_input_details()[0]
        scale, zero_point = input_details["quantization"]
        input_board = input_board / scale + zero_point

        set_input(self.interpreter, input_board.astype("int8"))
        self.interpreter.invoke()

        output_details = self.interpreter.get_output_details()[0]
        output_data = np.squeeze(self.interpreter.tensor(output_details["index"])())
        scale, zero_point = output_details["quantization"]
        # print(scale, zero_point)
        logits = scale * (output_data - zero_point)

        output_details = self.interpreter.get_output_details()[1]
        output_data = np.squeeze(self.interpreter.tensor(output_details["index"])())
        scale, zero_point = output_details["quantization"]
        value = scale * (output_data - zero_point)

        return (logits, np.array([value]))


def input_details(interpreter, key):
    """Returns input details by specified key."""
    return interpreter.get_input_details()[0][key]


def input_size(interpreter):
    """Returns input image size as (width, height) tuple."""
    _, height, width, _ = input_details(interpreter, "shape")
    return width, height


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = input_details(interpreter, "index")
    return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter, dequantize=True):
    """Returns output tensor of classification model.

    Integer output tensor is dequantized by default.

    Args:
      interpreter: tflite.Interpreter;
      dequantize: bool; whether to dequantize integer output tensor.

    Returns:
      Output tensor as numpy array.
    """
    output_details = interpreter.get_output_details()[0]
    output_data = np.squeeze(interpreter.tensor(output_details["index"])())

    if dequantize and np.issubdtype(output_details["dtype"], np.integer):
        scale, zero_point = output_details["quantization"]
        return scale * (output_data - zero_point)

    return output_data


def set_input(interpreter, data):
    """Copies data to input tensor."""
    input_tensor(interpreter)[:, :] = data
