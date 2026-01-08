from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO

app = Flask(__name__)

model = load_model("model/deepfake_model.h5")
yolo = YOLO("yolo/yolov8-face.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        faces = yolo(img)
        for face in faces[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, face)
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224)) / 255.0
            crop = np.expand_dims(crop, axis=0)
            
            pred = model.predict(crop)
            result = "Fake" if pred > 0.5 else "Real"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
