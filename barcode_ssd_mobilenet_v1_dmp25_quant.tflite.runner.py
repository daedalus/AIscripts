import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

# Load the TFLite model
model_path = "barcode_ssd_mobilenet_v1_dmp25_quant.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get model input size
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])

def preprocess_frame(frame):
    """Prepares a frame for inference."""
    img = cv2.resize(frame, input_size)  # Resize
    img = np.expand_dims(img, axis=0).astype(np.uint8)  # Add batch dim & convert
    return img

def run_inference(frame):
    """Runs inference on a frame and returns results."""
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

def draw_boxes(frame, output_data, threshold=0.5):
    """Draws bounding boxes on the live camera feed."""
    h, w, _ = frame.shape
    boxes, scores, _, num_detections = output_data

    for i in range(int(num_detections[0])):
        if scores[0][i] > threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{scores[0][i]:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    output_data = run_inference(frame_rgb)  # Run model
    frame = draw_boxes(frame, output_data)  # Overlay boxes

    cv2.imshow("Barcode Detector", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

