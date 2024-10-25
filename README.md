# Autonomous Drone Project - Object Detection Using OpenCV and ONNX

## Overview
This project is part of an autonomous drone system for real-time object detection and navigation. In this module, we demonstrate object detection using a pre-trained YOLOv5 model, which has been exported to ONNX format for optimized inference using OpenCV and ONNX Runtime.


https://github.com/user-attachments/assets/256bc449-6d8f-4b31-a971-4335b83dabce


Performance Comparison
ImplementationAverage FPSInference Time (ms)ImprovementCPU Only12201.4BaselineCUDA2575.22.7x fasterTensorRT1327.5726.6x faster
Detailed Performance Metrics
CPU Implementation

Average FPS: 12
Inference Time: 201.4ms
Suitable for basic testing and development

CUDA Implementation

Average FPS: 25
Inference Time: 75.2ms
62.65% improvement over CPU
Good for real-time applications

TensorRT Implementation

Average FPS: 132
Inference Time: 7.57ms
96.24% improvement over CPU
Optimal for high-performance requirements
Success Rate: 100%
Standard Deviation: 4.90ms

Requirements

Python 3.8+
OpenCV
CUDA Toolkit 11.4+
TensorRT 8.0+
NVIDIA GPU with CUDA support
Pre-trained YOLOv5 model

Installation

Clone the repository:

bashCopygit clone https://github.com/paulisure/AutonomousDrone-EmbeddedAI.git
cd AutonomousDrone-EmbeddedAI/object_detection

Install dependencies:

bashCopypip install -r requirements.txt

Install CUDA and TensorRT (for accelerated versions)

Running Different Implementations
CPU Version
bashCopypython object_detection_cpu.py
CUDA Version
bashCopypython object_detection_cuda.py
TensorRT Version
bashCopy# First, convert model to TensorRT
python convert_to_tensorrt.py

# Then run detection
python object_detection_tensorrt.py
Implementation Details
CPU Implementation

Basic implementation using OpenCV and ONNX Runtime
No hardware acceleration
Suitable for development and testing

CUDA Implementation

Utilizes NVIDIA CUDA for GPU acceleration
Significant performance improvement over CPU
Good balance of performance and implementation complexity

TensorRT Implementation

Highest performance implementation
Optimized for NVIDIA GPUs
Features:

FP16 precision support
Optimized layer fusion
Efficient memory management
Asynchronous inference
Batch processing capability



Project Structure
Copyobject_detection/
├── models/
│   ├── yolov5s.onnx
│   └── yolov5s.trt
├── src/
│   ├── object_detection_cpu.py
│   ├── object_detection_cuda.py
│   └── object_detection_tensorrt.py
├── utils/
│   └── convert_to_tensorrt.py
└── README.md

Future Improvements

Multi-GPU support
Batch processing optimization
Model quantization
Docker containerization

License
This project is licensed under the MIT License - see the LICENSE file for details.

