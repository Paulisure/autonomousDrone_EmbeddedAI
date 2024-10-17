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
