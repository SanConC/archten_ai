import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype(np.float32)
    img_array = (img_array / 127.5) - 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path):
    input_data = preprocess_image(image_path)
    prediction = model.predict(input_data, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_label = labels[predicted_index]
    predicted_score = float(prediction[predicted_index])

    result = {
        "label": predicted_label,
        "score": predicted_score,
        "scores": {
            labels[i]: float(prediction[i]) for i in range(len(labels))
        }
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict(image_path)
    print(json.dumps(result, ensure_ascii=False))