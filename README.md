# Autonomous Drone Project - Object Detection Using OpenCV and ONNX

## Overview
This project is part of an autonomous drone system for real-time object detection and navigation. In this module, we demonstrate object detection using a pre-trained YOLOv5 model, which has been exported to ONNX format for optimized inference using OpenCV and ONNX Runtime.

## Requirements
- Python 3.8+
- OpenCV
- ONNX Runtime
-Pre-trained YOLOv5 model (in ONNX format)

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/paulisure/AutonomousDrone-EmbeddedAI.git
    cd AutonomousDrone-EmbeddedAI/object_detection
    ```

2. Get all requirements:

Open your terminal on Windows and install the necessary libraries:
    ```bash
    pip install opencv-python onnx onnxruntime numpy
    ```

Visit YOLO's GitHub below. We'll be converting this using ONNX:
https://github.com/ultralytics/yolov5/blob/master/README.md

If you don't already have YOLO installed, follow these quick steps:

cd to/your/desired/directory
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install



3. Export YOLO model to ONNX format.

terminal:
python export.py --weights yolov5s.pt --img 640 --batch 1 --device 0 --include onnx

Expected output:
C:\Github_Projects\yolov5>python export.py --weights yolov5s.pt --img 640 --batch 1 --device 0 --include onnx
export: data=C:\Github_Projects\yolov5\data\coco128.yaml, weights=['yolov5s.pt'], imgsz=[640], batch_size=1, device=0, half=False, inplace=False, keras=False, optimize=False, int8=False, per_tensor=False, dynamic=False, simplify=False, mlmodel=False, opset=17, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5  v7.0-374-g94a62456 Python-3.12.2 torch-2.3.1 CUDA:0 (NVIDIA GeForce RTX 4080 Laptop GPU, 12282MiB)

Fusing layers...
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs

PyTorch: starting from yolov5s.pt with output shape (1, 25200, 85) (14.1 MB)

ONNX: starting export with onnx 1.17.0...
ONNX: export success  0.6s, saved as yolov5s.onnx (28.0 MB)

Export complete (1.5s)
Results saved to C:\Github_Projects\yolov5
Detect:          python detect.py --weights yolov5s.onnx
Validate:        python val.py --weights yolov5s.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.onnx')
Visualize:       https://netron.app

Successfully exported YOLO model to ONNX format.

Step 4:
Now, let's write the object detection script with OpenCV and ONNX.

In object_detection/ directory, open the object_detection_opencv_onnx.py file to make any necessary adjustments to meet your requirements such as the yolo.onnx file path, and adjusting functions preprocess and postprocess in order to match your requirements.

Once updated, go back to the terminal, and change directory to object_detection. 

Then run the python file:
    ```bash
    python object_detection_opencv_onnx.py
    ```

## Output
The script will use your webcam (or a video feed) to detect objects in real-time and display them with bounding boxes and labels.
