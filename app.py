from flask import Flask, request, jsonify
from PIL import Image
import torch
import time

app = Flask(__name__)

# โหลดโมเดล yolov10n.pt จาก local
model = torch.load("mix(320x160).pt", map_location="cpu")
model.eval()

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files['image'].stream).convert("RGB")
    t1 = time.time()
    results = model(img)
    t2 = time.time()

    return jsonify({
        "inference_time_ms": round((t2 - t1) * 1000, 2),
        "results": results.pandas().xyxy[0].to_dict(orient="records")
    })


