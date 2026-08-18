import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "model/best_efficientnet_focal.keras"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

IMAGE_PATH = r"D:\ajay\Downloads\DeepFER\images\873.jpg"

print("Loading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded!")

# --------------------------------
# Load original dataset image
# --------------------------------

image = Image.open(IMAGE_PATH)

print("Original image:", image.size, image.mode)

# Dataset is grayscale 48x48
image = image.convert("L")

# Resize exactly like train_generator
image = image.resize((224, 224))

# Convert grayscale -> RGB
image = image.convert("RGB")

# Convert to numpy
image = np.array(image, dtype=np.float32)

# IMPORTANT:
# Do NOT divide by 255.
# EfficientNet preprocessing is built into the model.
image = np.expand_dims(image, axis=0)

print("Model input shape:", image.shape)
print("Pixel range:", image.min(), "to", image.max())

# --------------------------------
# Prediction
# --------------------------------

prediction = model.predict(image, verbose=0)[0]

print("\nEmotion probabilities:")

for emotion, probability in zip(EMOTIONS, prediction):
    print(f"{emotion:<10}: {probability * 100:.2f}%")

predicted_index = np.argmax(prediction)

print("\nPredicted Emotion:", EMOTIONS[predicted_index])
print(
    "Confidence:",
    f"{prediction[predicted_index] * 100:.2f}%"
)