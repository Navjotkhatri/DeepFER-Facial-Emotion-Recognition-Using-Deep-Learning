# DeepFER - Facial Emotion Recognition Using CNN & Transfer Learning

DeepFER is a deep learning-based Facial Emotion Recognition (FER) system that classifies human facial expressions into seven different emotion categories using **EfficientNetB0 Transfer Learning**.

The project includes model training, evaluation, image-based prediction, validation testing, and a foundation for real-time emotion recognition.

---

## 📌 Project Overview

Facial Emotion Recognition is a computer vision problem where a machine learning model identifies the emotional state expressed through a person's facial features.

In this project, a transfer learning approach was implemented using **EfficientNetB0** as the feature extraction backbone.

The model classifies facial expressions into:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The original dataset contains **48 × 48 grayscale facial images**.

The images were converted into the required input format for EfficientNetB0 during preprocessing.

---

## 🎯 Project Objectives

The main objectives of DeepFER are:

1. Develop a facial emotion recognition model using deep learning.
2. Use transfer learning to improve feature extraction.
3. Classify facial expressions into seven emotion categories.
4. Evaluate model performance using accuracy, precision, recall, and F1-score.
5. Build an image-based prediction system.
6. Test the trained model on validation images.
7. Prepare the system for real-time emotion recognition.
8. Optimize the model for practical deployment.

---

## 🧠 Emotion Classes

The model uses the following class mapping:

```python
{
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}
