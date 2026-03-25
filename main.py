from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

app = FastAPI(title="ArchTen AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego lo restringimos
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
# Normalmente algo como (None, 224, 224, 3)
IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")

    # Resize con padding estilo Teachable Machine
    size = (IMG_WIDTH, IMG_HEIGHT)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)

    # Preprocesamiento típico de export Teachable Machine:
    # (image / 127.5) - 1
    normalized_image_array = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data

@app.get("/")
def root():
    return {"success": True, "message": "ArchTen AI running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        image = Image.open(io.BytesIO(contents))
        processed = preprocess_image(image)

        prediction = model.predict(processed, verbose=0)
        index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][index])

        raw_label = class_names[index]

        # Limpieza de label por si viene como "0 POTENCIAL ARQUEOLOGICO"
        clean_label = raw_label
        if " " in raw_label:
            clean_label = raw_label.split(" ", 1)[1].strip()

        return {
            "success": True,
            "label": clean_label,
            "raw_label": raw_label,
            "class_index": index,
            "confidence": round(confidence, 4)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")