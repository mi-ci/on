from flask import Flask, request, jsonify
import numpy as np
import cv2
import io
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('ep100.pt')
a = model.names
@app.route('/predict', methods=['POST'])
def upload_file():
    file = request.files['frame']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLOv8 model inference to detect objects
    results = model(image)

    # Extract bounding boxes and draw them on the image
    bounding_boxes = []
    for result in results[0].boxes:
        xmin, ymin, xmax, ymax = result.xyxy[0]
        confidence = result.conf[0]
        if confidence > 0.2:  # Confidence threshold
            bounding_boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    # Encode image with bounding boxes to base64
    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    em_li = []
    for i in list(results[0].boxes.cls.detach().numpy()):
        em_li.append(a[i])

    response = {
        'image': encoded_image,
        'classifications': em_li
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
