import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURATION
# ============================================================

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

# Your latest validation result
VALIDATION_ACCURACY = 45.0


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .emotion-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        background-color: #f5f7fa;
        margin-top: 20px;
    }

    .emotion-name {
        font-size: 32px;
        font-weight: 700;
    }

    .confidence {
        font-size: 20px;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">😊 DeepFER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Facial Emotion Recognition using EfficientNetB0'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_deepfer_model():

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    return model


try:

    model = load_deepfer_model()

except Exception as e:

    st.error(
        "Could not load the DeepFER model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        "**Architecture:** EfficientNetB0"
    )

    st.write(
        "**Classes:** 7 emotions"
    )

    st.write(
        "**Original image:** 48 × 48 grayscale"
    )

    st.write(
        "**Model input:** 224 × 224 RGB"
    )

    st.write(
        "**Validation accuracy:** ~45%"
    )

    st.divider()

    st.write("### Emotion Classes")

    for emotion in EMOTIONS:
        st.write(f"• {emotion}")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🖼️ Image Prediction",
        "📷 Camera",
        "📊 Model Analysis"
    ]
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(image):

    # --------------------------------
    # Grayscale
    # --------------------------------

    image = image.convert("L")

    # --------------------------------
    # Resize
    # --------------------------------

    image = image.resize(
        IMAGE_SIZE
    )

    # --------------------------------
    # Grayscale → RGB
    # --------------------------------

    image = image.convert("RGB")

    # --------------------------------
    # NumPy
    # --------------------------------

    image = np.array(
        image,
        dtype=np.float32
    )

    # --------------------------------
    # Batch dimension
    # --------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------
    # Prediction
    # --------------------------------

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        prediction
    )

    emotion = EMOTIONS[
        predicted_index
    ]

    confidence = prediction[
        predicted_index
    ]

    return (
        emotion,
        confidence,
        prediction
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    emotion,
    confidence,
    prediction
):

    st.markdown(
        f"""
        <div class="emotion-box">

        <div class="emotion-name">
        {emotion}
        </div>

        <div class="confidence">
        Confidence: {confidence * 100:.2f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "📊 Emotion Probabilities"
    )

    data = pd.DataFrame(
        {
            "Emotion": EMOTIONS,
            "Probability": prediction * 100
        }
    )

    data = data.sort_values(
        "Probability",
        ascending=False
    )

    for _, row in data.iterrows():

        emotion_name = row["Emotion"]
        probability = row["Probability"]

        st.write(
            f"**{emotion_name}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            int(
                min(
                    probability,
                    100
                )
            )
        )


# ============================================================
# TAB 1 — IMAGE PREDICTION
# ============================================================

with tab1:

    st.header(
        "Upload an Image"
    )

    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.image(
                image,
                caption="Input Image",
                use_container_width=True
            )

        with col2:

            if st.button(
                "🔍 Predict Emotion",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing facial expression..."
                ):

                    emotion, confidence, prediction = (
                        predict_emotion(image)
                    )

                display_results(
                    emotion,
                    confidence,
                    prediction
                )


# ============================================================
# TAB 2 — CAMERA
# ============================================================

with tab2:

    st.header(
        "📷 Camera Prediction"
    )

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image:

        image = Image.open(
            camera_image
        )

        st.image(
            image,
            caption="Camera Image",
            use_container_width=True
        )

        with st.spinner(
            "Analyzing facial expression..."
        ):

            emotion, confidence, prediction = (
                predict_emotion(image)
            )

        display_results(
            emotion,
            confidence,
            prediction
        )


# ============================================================
# TAB 3 — MODEL ANALYSIS
# ============================================================

with tab3:

    st.header(
        "📊 Model Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "EfficientNetB0"
        )

    with col2:

        st.metric(
            "Emotion Classes",
            "7"
        )

    with col3:

        st.metric(
            "Validation Accuracy",
            "~45%"
        )

    st.divider()

    st.subheader(
        "Emotion Classes"
    )

    class_data = pd.DataFrame(
        {
            "Class": EMOTIONS
        }
    )

    st.dataframe(
        class_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Preprocessing Pipeline"
    )

    st.code(
        """
48 × 48 Grayscale Image
          ↓
Resize to 224 × 224
          ↓
Convert Grayscale → RGB
          ↓
EfficientNetB0
          ↓
7-Class Softmax Prediction
        """,
        language="text"
    )

    st.info(
        "The model was trained on facial emotion "
        "images with seven emotion classes."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DeepFER — Facial Emotion Recognition "
    "using Transfer Learning"
)