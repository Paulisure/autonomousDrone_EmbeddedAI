import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time


LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "TVmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def load_engine(trt_engine_path):
    """Load the TensorRT engine"""
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, context):
    """Allocate buffers for input and output"""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(binding)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(tensor_shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem, 'tensor_name': tensor_name})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'tensor_name': tensor_name})
    
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    """
    Execute inference with proper error handling
    """
    try:
        # Transfer input data to device
        for inp in inputs:
            if not inp['host'].flags['C_CONTIGUOUS']:
                inp['host'] = np.ascontiguousarray(inp['host'])
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            context.set_tensor_address(inp['tensor_name'], inp['device'])
        
        # Set output tensor addresses
        for out in outputs:
            context.set_tensor_address(out['tensor_name'], out['device'])
        
        # Run inference
        context.execute_async_v3(stream_handle=stream.handle)
        
        # Transfer predictions back
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        
        stream.synchronize()
        
    except Exception as e:
        print(f"Inference error: {str(e)}")
        raise

def preprocess(image):
    """
    Preprocess image with guaranteed contiguous output
    """
    input_height, input_width = 640, 640
    h, w = image.shape[:2]
    
    # Calculate scale while maintaining aspect ratio
    scale = min(input_height/h, input_width/w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create black canvas
    canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_x = (input_width - new_w) // 2
    pad_y = (input_height - new_h) // 2
    
    # Place resized image on canvas
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    # Convert to float32, normalize, and ensure memory is contiguous
    img = np.ascontiguousarray(canvas.astype(np.float32) / 255.0)
    
    # Transpose and add batch dimension (ensuring contiguous memory)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = np.ascontiguousarray(np.expand_dims(img, axis=0))
    
    return img, {
        'original_size': (w, h),
        'scale': scale,
        'padding': (pad_x, pad_y)
    }

def postprocess(output, original_img, preprocess_info):
    """Postprocess TensorRT output"""
    original_w, original_h = preprocess_info['original_size']
    output = output.reshape((1, 25200, 85))
    
    boxes = []
    scores = []
    class_ids = []
    
    # Process detections
    for detection in output[0]:
        confidence = float(detection[4])
        
        if confidence > 0.25:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = float(class_scores[class_id])
            score = confidence * class_score
            
            if score > 0.25:
                # Get box coordinates
                box = detection[:4]
                box = box / 640.0  # Normalize by input size
                
                # Convert to image coordinates
                x_center = box[0] * original_w
                y_center = box[1] * original_h
                width = box[2] * original_w
                height = box[3] * original_h
                
                # Calculate corners
                x1 = int(max(0, x_center - width/2))
                y1 = int(max(0, y_center - height/2))
                x2 = int(min(original_w, x_center + width/2))
                y2 = int(min(original_h, y_center + height/2))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_ids.append(class_id)
    
    # Apply NMS
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        if len(indices) > 0:
            indices = indices.flatten()
            
            for idx in indices:
                box = boxes[idx]
                score = scores[idx]
                class_id = class_ids[idx]
                
                # Draw detection
                cv2.rectangle(original_img,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 255, 0), 2)
                
                # Add label
                label = f"{LABELS[class_id]}: {score:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                y_label = max(box[1], label_h + 10)
                cv2.rectangle(original_img,
                            (box[0], y_label - label_h - baseline - 5),
                            (box[0] + label_w + 5, y_label),
                            (0, 255, 0),
                            cv2.FILLED)
                
                cv2.putText(original_img,
                           label,
                           (box[0], y_label - 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 0, 0),
                           1)
    
    return original_img

def main():
    try:
        print("Loading TensorRT engine...")
        engine = load_engine("yolov5s.trt")
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine, context)
        
        # Initialize metrics
        total_inference_time = 0
        total_frames = 0
        inference_times = []
        successful_frames = 0
        
        print("Starting video capture...")
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            try:
                # Start timing before preprocessing
                start_time = time.time()
                
                # Preprocess
                preprocessed_img, preprocess_info = preprocess(frame)
                
                # Ensure input is contiguous
                if not preprocessed_img.flags['C_CONTIGUOUS']:
                    preprocessed_img = np.ascontiguousarray(preprocessed_img)
                
                # Copy to input buffer
                inputs[0]['host'][:] = preprocessed_img.ravel()
                
                # Inference timing
                inference_start = time.time()
                do_inference(context, bindings, inputs, outputs, stream)
                inference_time = time.time() - inference_start
                
                inference_times.append(inference_time * 1000)  # Convert to milliseconds
                total_inference_time += inference_time
                successful_frames += 1
                
                # Postprocess
                result_frame = postprocess(outputs[0]['host'], frame, preprocess_info)
                
                # Calculate FPS (including pre/post processing)
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Display FPS and inference time
                cv2.putText(result_frame,
                           f"FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           (0, 255, 0),
                           2)
                
                cv2.imshow("TensorRT YOLOv5 Detection", result_frame)
                
            except Exception as e:
                print(f"Error processing frame {total_frames}: {str(e)}")
                continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Print detailed metrics only if we have data
        if successful_frames > 0 and inference_times:
            print("\nPerformance Metrics:")
            print(f"Total frames attempted: {total_frames}")
            print(f"Successfully processed frames: {successful_frames}")
            print(f"Success rate: {(successful_frames/total_frames)*100:.1f}%")
            
            # Calculate statistics safely
            avg_inference = np.mean(inference_times)
            std_inference = np.std(inference_times) if len(inference_times) > 1 else 0
            min_inference = np.min(inference_times)
            max_inference = np.max(inference_times)
            
            print(f"Average inference time: {avg_inference:.2f}ms")
            print(f"Std dev inference time: {std_inference:.2f}ms")
            print(f"Min inference time: {min_inference:.2f}ms")
            print(f"Max inference time: {max_inference:.2f}ms")
            print(f"Average FPS: {successful_frames/total_inference_time:.1f}")
        else:
            print("\nNo frames were successfully processed")

if __name__ == "__main__":
    main()