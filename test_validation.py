import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# SETTINGS
# =========================

MODEL_PATH = "model/best_efficientnet_focal.keras"

VALIDATION_DIR = r"D:\ajay\Downloads\DeepFER\Face Emotion Recognition Dataset\images\validation"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!\n")


# =========================
# FUNCTION TO PREDICT IMAGE
# =========================

def predict_image(image_path):

    image = Image.open(image_path).convert("L")

    # 48x48 -> 224x224
    image = image.resize((224, 224))

    # Grayscale -> RGB
    image = image.convert("RGB")

    image = np.array(image, dtype=np.float32)

    # IMPORTANT:
    # Do NOT divide by 255.
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    predicted_emotion = EMOTIONS[predicted_index]

    confidence = prediction[predicted_index] * 100

    return predicted_emotion, confidence


# =========================
# TEST ONE IMAGE PER CLASS
# =========================

print("=" * 70)
print("VALIDATION DATASET TEST")
print("=" * 70)

correct = 0
total = 0


for emotion in EMOTIONS:

    folder = os.path.join(
        VALIDATION_DIR,
        emotion
    )

    if not os.path.exists(folder):

        print(f"{emotion:<10} | Folder not found")

        continue

    # Find images
    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if len(images) == 0:

        print(f"{emotion:<10} | No images found")

        continue

    # Take first image
    image_file = images[0]

    image_path = os.path.join(
        folder,
        image_file
    )

    predicted, confidence = predict_image(
        image_path
    )

    is_correct = predicted.lower() == emotion

    if is_correct:
        correct += 1
        result = "CORRECT"
    else:
        result = "WRONG"

    total += 1

    print(
        f"{emotion:<10} | "
        f"Predicted: {predicted:<9} | "
        f"Confidence: {confidence:6.2f}% | "
        f"{result}"
    )


# =========================
# SUMMARY
# =========================

print("\n" + "=" * 70)

if total > 0:

    accuracy = (
        correct / total
    ) * 100

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Sample Accuracy: {accuracy:.2f}%"
    )
