import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

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


# Load the TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def load_engine(trt_engine_path):
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, context):
    # Allocate host and device buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Use engine.num_io_tensors to get the number of tensors (bindings)
    for binding in range(engine.num_io_tensors):
        # Retrieve the name of the tensor
        tensor_name = engine.get_tensor_name(binding)

        # Use the tensor name to get its shape
        binding_shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the allocated memory to the lists
        bindings.append(int(device_mem))

        # Use get_tensor_mode to check if it's an input or output tensor
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem, 'tensor_name': tensor_name})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'tensor_name': tensor_name})

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Ensure that the inputs are contiguous before copying to GPU
    for inp in inputs:
        inp['host'] = np.ascontiguousarray(inp['host'])  # Make sure it's contiguous
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

    # Set tensor addresses for inputs and outputs
    for inp in inputs:
        tensor_name = inp.get('tensor_name')
        context.set_tensor_address(tensor_name, inp['device'])

    for out in outputs:
        tensor_name = out.get('tensor_name')
        context.set_tensor_address(tensor_name, out['device'])

    # Run inference using execute_async_v3
    context.execute_async_v3(stream_handle=stream.handle)

    # Transfer predictions back from the GPU
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

    # Synchronize the stream
    stream.synchronize()


def preprocess(image):
    img = cv2.resize(image, (640, 640))  # Resize to YOLO input size
    img = img.transpose(2, 0, 1)  # Change data format from HWC to CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)  # Add batch dimension
    return img / 255.0  # Normalize to [0, 1]



def postprocess(output, image):
    h, w, _ = image.shape

    # Reshape output to (1, 25200, 85)
    output = output.reshape(1, 25200, 85)
    detections = output[0]

    for det in detections:
        conf = det[4]
        if conf > 0.5:
            x1, y1, x2, y2 = det[:4]
            
            # Convert normalized coordinates to pixel values
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            class_id = int(np.argmax(det[5:]))
            label = LABELS[class_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image





if __name__ == "__main__":
    trt_engine_path = "yolov5s.trt"  # Path to TensorRT engine

    # Load the TensorRT engine
    engine = load_engine(trt_engine_path)
    context = engine.create_execution_context()

    # Allocate memory for inputs/outputs
    inputs, outputs, bindings, stream = allocate_buffers(engine, context)

    # Set up the video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the input image
        inputs[0]['host'] = preprocess(frame)

        # Perform inference
        do_inference(context, bindings, inputs, outputs, stream)

        # Check output shape and values
        print("Output shape:", outputs[0]['host'].shape)
        print("Output values:", outputs[0]['host'])
        print("Output min:", np.min(outputs[0]['host']))
        print("Output max:", np.max(outputs[0]['host']))
        print("Output mean:", np.mean(outputs[0]['host']))

        # Postprocess the output and display the result
        result_frame = postprocess(outputs[0]['host'], frame)
        if result_frame is None or result_frame.size == 0:
            print("Invalid frame generated!")
        else:
            cv2.imshow("TensorRT Object Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
