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
        print(f"Loading EdgeTPU model from {model_file}")
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

        input_details = self.interpreter.get_input_details()
        self.input_quantization = input_details[0]["quantization"]
        self.input_index = input_details[0]["index"]

        output_details = self.interpreter.get_output_details()
        self.logits_index = output_details[0]["index"]
        self.logits_quantization = output_details[0]["quantization"]

        self.value_index = output_details[1]["index"]
        self.value_quantization = output_details[1]["quantization"]

    def predict(self, input_board):
        scale, zero_point = self.input_quantization
        input_board = input_board / scale + zero_point

        self.interpreter.tensor(self.input_index)()[0][:,:] = input_board.astype("int8")
        self.interpreter.invoke()

        output_data = np.squeeze(self.interpreter.tensor(self.logits_index)())
        scale, zero_point = self.logits_quantization
        # print(scale, zero_point)
        logits = scale * (output_data - zero_point)

        output_data = np.squeeze(self.interpreter.tensor(self.value_index)())
        scale, zero_point = self.value_quantization
        value = scale * (output_data - zero_point)

        return (logits, value)


if __name__ == "__main__":
    from time import perf_counter

    N = 5000

    model = EdgeTpuModel("models/tflite-128x19_edgetpu.tflite")

    data = np.ones((1, 8, 8, 19)).astype("int8")

    p_start = perf_counter()
    for i in range(N):
        _ = model.predict(data)
    p_end = perf_counter()

    print(f"{N} predictions in {p_end-p_start}s = {N/(p_end-p_start)} 1/s")
