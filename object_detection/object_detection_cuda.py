import cv2
import onnxruntime as ort
import numpy as np
import time

# Load the ONNX model with CUDA support
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_model_path = 'yolov5s.onnx'
session = ort.InferenceSession(onnx_model_path, providers=providers)

# Load the labels for YOLOv5 model
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', ...]  # Add all YOLO labels

# Set up the video capture (use webcam or video file)
cap = cv2.VideoCapture(0)

# Preprocess the image for YOLOv5
def preprocess(image):
    img = cv2.resize(image, (640, 640))  # Resize to YOLO input size
    img = img.transpose(2, 0, 1)  # Change data format from HWC to CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)  # Add batch dimension
    return img / 255.0  # Normalize to [0, 1]

# Postprocess the output to show bounding boxes
def postprocess(output, image):
    h, w, _ = image.shape

    # Assuming the output contains (1, 1, 25200, 85) where N = 25200 is the number of detections
    # and each detection has 85 values, including bounding box, objectness score, and class scores.
    detections = output[0]  # Output is a list, get the first element (the array)
    
    # Loop through each detection
    for det in detections[0]:
        conf = det[4]  # Confidence score
        
        # Check if conf is a scalar or an array
        if isinstance(conf, np.ndarray):
            conf = conf.item()  # Convert array to scalar if necessary
        
        # Only consider detections with a confidence score above the threshold
        if conf > 0.5:  # Confidence threshold
            
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, det[:4])

            # Extract the class label index and map it to the corresponding label
            class_id = int(np.argmax(det[5:]))  # Assuming class scores start at index 5
            print(f"Class ID: {class_id}, Number of labels: {len(LABELS)}")  # Debugging line

            # Check if the class_id is within the bounds of the LABELS list
            if class_id < len(LABELS):
                label = LABELS[class_id]
                # Draw the bounding box and label on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print(f"Invalid class ID: {class_id}")  # Handle invalid class ID

    return image

# Initialize variables to track inference time
total_inference_time = 0
num_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Preprocess the frame for inference
    input_tensor = preprocess(frame)

    # Run the ONNX model (using CUDA if specified)
    output = session.run(None, {'images': input_tensor})

    # Postprocess the output and display the result
    result_frame = postprocess(output, frame)

    # Calculate the inference time for this frame
    inference_time = time.time() - start_time
    total_inference_time += inference_time
    num_frames += 1

    # Display the current frame's inference time
    cv2.putText(result_frame, f"Inference Time: {inference_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the real-time video feed
    cv2.imshow("CUDA Accelerated Object Detection", result_frame)

    # Stop after 100 frames for benchmarking
    if num_frames == 100:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate and print the average inference time
avg_inference_time = total_inference_time / num_frames
print(f"Average Inference Time over {num_frames} frames: {avg_inference_time:.4f} seconds")

