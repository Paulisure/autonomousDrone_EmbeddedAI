import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_model_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision

    return builder.build_engine(network, config)


    # Load the ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Build the serialized engine
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the engine.")
        return None

    # Deserialize the engine for inference
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine



def save_engine(engine, engine_file_path):
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to: {engine_file_path}")

def convert_onnx_to_trt(onnx_model_path, trt_engine_path):
    engine = build_engine(onnx_model_path)
    if engine:
        save_engine(engine, trt_engine_path)

if __name__ == "__main__":
    onnx_model_path = "yolov5s.onnx"  # Path to your ONNX model
    trt_engine_path = "yolov5s.trt"   # Output path for TensorRT engine
    convert_onnx_to_trt(onnx_model_path, trt_engine_path)
