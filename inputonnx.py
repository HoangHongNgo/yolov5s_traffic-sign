import onnxruntime

# Load the ONNX model
model = onnxruntime.InferenceSession("last.onnx")

# Get the inputs of the ONNX model
inputs = model.get_outputs()

# Print the name of each input
for input in inputs:
    print(input.name)