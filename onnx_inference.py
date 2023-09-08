import time
import numpy as np
import torch
import onnxruntime

onnx_file = "loop.onnx"
sess = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

texts_len = 15
dummy_input_1 = torch.FloatTensor(1, texts_len, 256).numpy()

output_names = ['output']
output = sess.run(output_names, {'texts': dummy_input_1})
print(output[0].shape)