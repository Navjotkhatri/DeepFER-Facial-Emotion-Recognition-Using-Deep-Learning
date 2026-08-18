from tensorflow.keras.models import load_model

MODEL_PATH = "model/best_efficientnet_focal.keras"

print("Loading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model Loaded Successfully!")
print()
model.summary()