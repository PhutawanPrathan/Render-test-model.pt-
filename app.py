from flask import Flask, request, jsonify
from PIL import Image
import torch
import time

# สำหรับให้ torch.load โหลด object ที่มี class จาก ultralytics
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

app = Flask(__name__)

# โหลดโมเดลจาก .pt (ต้องเป็นไฟล์ที่เชื่อถือได้)
model = torch.load("mix(320x160).pt", map_location="cpu", weights_only=False)
model.eval()

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files['image'].stream).convert("RGB")
    
    t1 = time.time()
    results = model(img)  # สำหรับ Ultralytics model
    t2 = time.time()

    # ดึงผลลัพธ์แบบ pandas → dict
    try:
        pandas_result = results.pandas().xyxy[0].to_dict(orient="records")
    except:
        pandas_result = []

    return jsonify({
        "inference_time_ms": round((t2 - t1) * 1000, 2),
        "results": pandas_result
    })

# ให้ gunicorn รู้จักตัวแปร app
app = app
