"""
Diagnostic: run from the project root (the folder that contains `app` and `test_images`).
Put your own leaf photos in a folder called `my_leaves` (create it) to test them too.
    python check_model.py
"""
import glob
import json
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

MODEL = "app/trained_model/plant_disease_prediction_model.keras"
CLASSES = "app/class_indices.json"

# 1) Which model file is really being used?
size_mb = os.path.getsize(MODEL) / 1e6
print(f"Model file: {size_mb:.0f} MB, saved {time.ctime(os.path.getmtime(MODEL))}")
print("  -> the new MobileNetV2 model is about 10 MB; the OLD model was ~500 MB\n")

model = tf.keras.models.load_model(MODEL, compile=False)
has_rescaling = any(isinstance(l, tf.keras.layers.Rescaling) for l in model.layers)
print("Built-in Rescaling layer:", has_rescaling, "| outputs:", model.output_shape[-1])

names = {str(k): v for k, v in json.load(open(CLASSES)).items()}
print("Classes in json:", len(names), "| first:", names["0"], "| last:", names[str(len(names) - 1)], "\n")

EXPECTED = {  # test image file name -> correct class
    "test_apple_black_rot": "Apple___Black_rot",
    "test_blueberry_healthy": "Blueberry___healthy",
    "test_potato_early_blight": "Potato___Early_blight",
}


def predict(img, divisor):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224))
    x = np.expand_dims(np.array(img, dtype="float32") / divisor, 0)
    p = model.predict(x, verbose=0)[0]
    return [(names[str(i)], float(p[i]) * 100) for i in np.argsort(p)[::-1][:3]]


def show(title, img, divisor):
    print(title)
    for n, c in predict(img, divisor):
        print(f"    {c:5.1f}%  {n}")


# 2) Try BOTH pixel scalings on the 3 known test images
print("=== Which pixel scaling does this model need? ===")
scores = {}
for divisor in (1.0, 255.0):
    ok = 0
    print(f"\n--- pixels / {divisor:g} ---")
    for p in sorted(glob.glob("test_images/*")):
        key = os.path.splitext(os.path.basename(p))[0]
        top = predict(Image.open(p), divisor)[0]
        right = EXPECTED.get(key) == top[0]
        ok += right
        print(f"  {'OK ' if right else 'BAD'} {key}: {top[0]} ({top[1]:.1f}%)")
    scores[divisor] = ok
best = max(scores, key=scores.get)
print(f"\nRESULT: set PIXEL_DIVISOR_OVERRIDE = {best:g} in app/main.py "
      f"({scores[best]}/3 test images correct)\n")

# 3) Your own photos, using the best scaling
for p in sorted(glob.glob(os.path.join("my_leaves", "*"))):
    try:
        show(p, Image.open(p), best)
    except Exception as e:
        print(p, "skipped:", e)