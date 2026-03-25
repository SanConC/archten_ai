import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opcional (mejor para móviles):
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("archten_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Listo: archten_model.tflite creado")