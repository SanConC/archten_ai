from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

app = FastAPI(title="ArchTen AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y labels una sola vez al iniciar
model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# Detectar tamaño de entrada del modelo automáticamente
input_shape = model.input_shape
IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]

# Umbral mínimo de confianza
CONFIDENCE_THRESHOLD = 0.75


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")

    # Resize con padding estilo Teachable Machine
    size = (IMG_WIDTH, IMG_HEIGHT)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)

    # Normalización típica de Teachable Machine
    normalized_image_array = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data


@app.get("/")
def root():
    return {
        "success": True,
        "message": "ArchTen AI running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")

        processed = preprocess_image(image)

        prediction = model.predict(processed, verbose=0)
        scores = prediction[0]

        index = int(np.argmax(scores))
        confidence = float(scores[index])

        raw_label = class_names[index].strip()
        clean_label = raw_label

        final_label = clean_label
        if confidence < CONFIDENCE_THRESHOLD:
            final_label = "INCIERTO"

        return {
            "success": True,
            "label": final_label,
            "predicted_label": clean_label,
            "raw_label": raw_label,
            "class_index": index,
            "confidence": round(confidence, 4),
            "threshold": CONFIDENCE_THRESHOLD,
            "scores": [round(float(x), 4) for x in scores.tolist()],
            "labels": class_names
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {str(e)}"
        )