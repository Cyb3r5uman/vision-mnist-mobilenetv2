import argparse
from tensorflow import keras
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Predict MNIST digit using trained model")
    parser.add_argument("--model", required=True, help="Path to saved .keras model")
    parser.add_argument("--image", required=True, help="Path to image file")
    return parser.parse_args()

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((96, 96))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    args = parse_args()
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)

    img_array = preprocess_image(args.image)
    preds = model.predict(img_array)

    predicted_label = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    print(f"Predicted digit: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
