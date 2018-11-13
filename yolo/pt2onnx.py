import torch

from darknet import Darknet

model = Darknet('./yolov3-spp.cfg')
model.load_weights('./yolov3-spp.weights')
#
dummy = torch.Tensor(torch.randn(1, 3, 608, 608))
torch.onnx.export(model, dummy, 'yolov3.onnx')
