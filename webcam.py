import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# SETTINGS
# ==========================================

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

IMAGE_SIZE = (224, 224)

# Don't display predictions below this confidence
CONFIDENCE_THRESHOLD = 0.30


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading DeepFER model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# LOAD FACE DETECTOR
# ==========================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Face detector could not be loaded.")
    print("Cascade path:")
    print(cascade_path)
    exit()

print("Face detector loaded successfully!")


# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()

print("\nWebcam started.")
print("Press Q to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ======================================
    # PROCESS EACH FACE
    # ======================================

    for (x, y, w, h) in faces:

        # Crop face from grayscale image
        face = gray[
            y:y+h,
            x:x+w
        ]

        # Resize to model size
        face = cv2.resize(
            face,
            IMAGE_SIZE
        )

        # Grayscale -> RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_GRAY2RGB
        )

        # Convert to float32
        face_rgb = face_rgb.astype(
            np.float32
        )

        # Add batch dimension
        face_input = np.expand_dims(
            face_rgb,
            axis=0
        )

        # ==================================
        # PREDICTION
        # ==================================

        prediction = model.predict(
            face_input,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            prediction
        )

        confidence = prediction[
            predicted_index
        ]

        emotion = EMOTIONS[
            predicted_index
        ]

        # ==================================
        # DRAW FACE BOX
        # ==================================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # ==================================
        # DISPLAY PREDICTION
        # ==================================

        if confidence >= CONFIDENCE_THRESHOLD:

            label = (
                f"{emotion}: "
                f"{confidence * 100:.1f}%"
            )

        else:

            label = (
                f"Uncertain: "
                f"{confidence * 100:.1f}%"
            )

        # Background for text
        cv2.rectangle(
            frame,
            (x, y - 35),
            (x+w, y),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "DeepFER - Real Time Emotion Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("DeepFER stopped.")