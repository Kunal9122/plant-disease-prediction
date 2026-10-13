import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model (path is relative to the project root)
m = tf.keras.models.load_model(
    "app/trained_model/plant_disease_prediction_model.keras",
    compile=False,
)
print("Input shape:", m.input_shape)
m.summary()


def prep(path, scale):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype="float32")
    return np.expand_dims(arr / scale, 0)


test_images = [
    "test_images/test_apple_black_rot.JPG",
    "test_images/test_blueberry_healthy.jpg",
    "test_images/test_potato_early_blight.jpg",
]

for p in test_images:
    for scale in (255.0, 1.0):
        pr = m.predict(prep(p, scale), verbose=0)[0]
        print(f"{p} | scale={scale} | class={pr.argmax()} | confidence={pr.max():.3f}")