import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size (must match training settings)
IMG_SIZE = (224, 224)  # adjust if you used a different size

def predict_image(img_path):
    """Predict the class of an input image."""
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    img_path = input("Enter image path: ")
    predict_image(img_path)
