"""
First run in terminal:
    pip install flask flask-cors
    python app.py

Then open ai_geoguesser.html in the browser.
"""

import io
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Config
CHECKPOINT_PATH = "best_model_resnet50.pth"
CLASSES         = ["istanbul", "nyc", "tokyo"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load model once at startup
def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

print("Loading model…")
MODEL = build_model()
print("Model ready.")

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs  = torch.softmax(logits, dim=1)[0].tolist()

    name_map = {"istanbul": "Istanbul", "nyc": "New York City", "tokyo": "Tokyo"}
    probabilities = {name_map[cls]: round(prob, 4) for cls, prob in zip(CLASSES, probs)}

    predicted_cls  = CLASSES[probs.index(max(probs))]
    predicted_city = name_map[predicted_cls]
    confidence     = round(max(probs), 4)

    return jsonify({
        "predicted_city": predicted_city,
        "confidence":     confidence,
        "probabilities":  probabilities,
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
