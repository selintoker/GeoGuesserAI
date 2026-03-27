import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
try:
    import easyocr
    OCR_READER = easyocr.Reader(["en", "ja", "tr"], verbose=False)
except ImportError:
    OCR_READER = None

CHECKPOINT_PATH = "best_model_resnet50.pth"
CLASSES = ["istanbul", "nyc", "tokyo"] 
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image_path: str, model):
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    results = sorted(zip(CLASSES, probs.tolist()), key=lambda x: x[1], reverse=True)
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <image_path> [image_path ...]")
        sys.exit(1)

    model = build_model()

    for path in sys.argv[1:]:
        results = predict(path, model)
        top_city, top_conf = results[0]
        print(f"\n{path}")
        print(f"  Prediction: {top_city} ({top_conf:.2%})")
        for city, conf in results:
            bar = "█" * int(conf * 30)
            print(f"  {city:<12} {conf:.2%}  {bar}")
        if OCR_READER is not None:
            texts = OCR_READER.readtext(path, detail=0)
            if texts:
                print(f"  Detected text: {', '.join(texts)}")


if __name__ == "__main__":
    main()
