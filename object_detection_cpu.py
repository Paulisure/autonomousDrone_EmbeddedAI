import cv2
import onnxruntime as ort
import numpy as np
import time

# Load the ONNX model (using CPU only)
onnx_model_path = 'yolov5s.onnx'
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

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

    # Extract detections from the output (remove batch dimension)
    detections = output[0][0]  # (25200, 85) array
    
    for det in detections:
        # Extract bounding box center coordinates, width, height
        x_center, y_center, width, height = det[:4]
        
        # Convert center-based (x_center, y_center) format to top-left (x, y) and bottom-right (x2, y2)
        x = int((x_center - width / 2) * w)
        y = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # Extract confidence score (5th value in det array)
        conf = det[4]

        # Only proceed if confidence is above the threshold (0.5)
        if conf > 0.5:
            # Extract class probabilities and determine the class with the highest probability
            class_probs = det[5:]
            class_id = np.argmax(class_probs)
            label = LABELS[class_id]

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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

    # Run the ONNX model using CPU
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
    cv2.imshow("CPU Object Detection", result_frame)

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
