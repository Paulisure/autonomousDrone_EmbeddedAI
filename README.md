# Autonomous Drone Project - Real-time Object Detection with Hardware Acceleration

## Overview
This project demonstrates real-time object detection using YOLOv5 with various hardware acceleration methods. We implement and compare CPU, CUDA, and TensorRT approaches to achieve optimal performance for autonomous drone applications.

https://github.com/user-attachments/assets/256bc449-6d8f-4b31-a971-4335b83dabce


## Performance Comparison

| Implementation | Average FPS | Inference Time (ms) | Improvement |
|----------------|-------------|---------------------|-------------|
| CPU Only | 12 | 201.4 | Baseline |
| CUDA | 25 | 75.2 | 2.7x faster |
| TensorRT | 132 | 7.57 | 26.6x faster |

### Detailed Performance Metrics

#### CPU Implementation
- Average FPS: 12
- Inference Time: 201.4ms
- Suitable for basic testing and development

#### CUDA Implementation
- Average FPS: 25
- Inference Time: 75.2ms
- 62.65% improvement over CPU
- Good for real-time applications

#### TensorRT Implementation
- Average FPS: 132
- Inference Time: 7.57ms
- 96.24% improvement over CPU
- Optimal for high-performance requirements
- Success Rate: 100%
- Standard Deviation: 4.90ms

## Requirements
- Python 3.8+
- OpenCV
- CUDA Toolkit 11.4+
- TensorRT 8.0+
- NVIDIA GPU with CUDA support
- Pre-trained YOLOv5 model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/paulisure/AutonomousDrone-EmbeddedAI.git
cd AutonomousDrone-EmbeddedAI/object_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install CUDA and TensorRT (for accelerated versions)

## Running Different Implementations

### CPU Version
```bash
python object_detection_cpu.py
```

### CUDA Version
```bash
python object_detection_cuda.py
```

### TensorRT Version
```bash
# First, convert model to TensorRT
python convert_to_tensorrt.py

# Then run detection
python object_detection_tensorrt.py
```

## Implementation Details

### CPU Implementation
- Basic implementation using OpenCV and ONNX Runtime
- No hardware acceleration
- Suitable for development and testing

### CUDA Implementation
- Utilizes NVIDIA CUDA for GPU acceleration
- Significant performance improvement over CPU
- Good balance of performance and implementation complexity

### TensorRT Implementation
- Highest performance implementation
- Optimized for NVIDIA GPUs
- Features:
  - FP16 precision support
  - Optimized layer fusion
  - Efficient memory management
  - Asynchronous inference
  - Batch processing capability

## Project Structure
```
object_detection/
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
```

## Future Improvements
- Multi-GPU support
- Batch processing optimization
- Model quantization
- Docker containerization

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


