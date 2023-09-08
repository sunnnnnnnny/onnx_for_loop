import yaml
import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(256, 80)

    def forward(self, inp):
        lens = inp.shape[1]
        inp = self.linear1(inp).squeeze(0)
        predicted = torch.randint(1, 12, (lens,))
        out = torch.zeros(predicted.sum(), 80)
        print(out[0,:])
        loop_range = inp.size(0)
        start_idx = torch.LongTensor([0])
        for i in range(loop_range):
            vec = inp[i, :]
            expand_size = predicted[i]
            for j in range(expand_size):
                out[start_idx+j, :] = vec
            start_idx = start_idx + expand_size
        print(out[0,:])
        return out


model = Model()

input_names = ['texts']
output_names = ['output']

dynamic_axes = {
    "texts": [0, 1],
    "output": [0, 1],
}

dummy_input_1 = torch.FloatTensor(1, 12, 256)
print(dummy_input_1.shape)

# model = torch.jit.script(model)
# output = model(dummy_input_1)
# print(output.shape)


torch.onnx.export(model, args=(dummy_input_1), f="loop.onnx",
                  input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)