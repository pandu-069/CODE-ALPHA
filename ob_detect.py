import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Use 'yolov5s' for speed, 'yolov5m', 'yolov5l', or 'yolov5x' for more accuracy

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 is the default camera, change it if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img_rgb)

    # Parse results
    predictions = results.xyxy[0].numpy()  # Get predictions as NumPy array
    labels, confidences, boxes = [], [], []

    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        labels.append(int(cls))
        confidences.append(float(conf))
        boxes.append([int(x1), int(y1), int(x2), int(y2)])

    # Draw bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = model.names[labels[i]]
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Face Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
