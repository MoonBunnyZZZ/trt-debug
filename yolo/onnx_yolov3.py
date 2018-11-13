# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    DTYPE = trt.float32


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()


def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network,
                                                                                                 TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        return builder.build_cuda_engine(network)


def load_normalized_test_case(test_image, pagelocked_buffer):
    def normalize_image(image):
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(
            trt.nptype(ModelData.DTYPE)).ravel()
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image


def main():
    data_path, data_files = common.find_sample_data(
        description="Runs a ResNet50 network with a TensorRT inference engine.", subfolder="resnet50",
        find_files=["binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg", ModelData.MODEL_PATH,
                    "class_labels.txt"])
    test_images = data_files[0:3]
    onnx_model_file, labels_file = data_files[3:]
    labels = open(labels_file, 'r').read().split('\n')

    with build_engine_onnx(onnx_model_file) as engine:
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            test_image = random.choice(test_images)
            test_case = load_normalized_test_case(test_image, h_input)
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            pred = labels[np.argmax(h_output)]
            print("Recognized " + test_case + " as " + pred)


if __name__ == '__main__':
    # main()
    build_engine_onnx('./yolov3.onnx')