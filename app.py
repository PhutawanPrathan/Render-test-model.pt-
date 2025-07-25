from flask import Flask, request, jsonify
from PIL import Image
import torch
import io, time

app = Flask(__name__)

# Load YOLOv10n model (.pt)
model = torch.hub.load('ultralytics/ultralytics', 'yolov10n', pretrained=False)
model.load_state_dict(torch.load("mix(320x160).pt", map_location="cpu"))
model.eval()

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "no image"}), 400

    img_file = request.files['image']
    img = Image.open(img_file.stream).convert("RGB")

    t1 = time.time()
    results = model(img, verbose=False)
    t2 = time.time()

    pred = results[0].boxes.data.tolist() if hasattr(results[0], 'boxes') else []

    return jsonify({
        "inference_time_ms": round((t2 - t1) * 1000, 2),
        "boxes": pred
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
