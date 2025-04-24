# Driver Drowsiness Detection

## 🚗 Project Overview

This project uses computer vision and machine learning to detect driver drowsiness in real-time using a webcam. It leverages a pre-trained deep learning model (InceptionV3) to classify whether the driver's eyes are open or closed, triggering an alarm if the eyes remain closed for a specific duration.

### 📸 Features:
- Real-time webcam face detection
- Eye state classification (open/closed) using a deep learning model
- Alarm system that sounds when the driver is detected to be drowsy
- Automated phone call alert via Twilio API if drowsiness is detected for an extended period

---

## 🧠 Model Information

The eye detection model is based on a custom-trained neural network using the **InceptionV3** architecture, fine-tuned on the [MRL Eye Dataset](https://www.kaggle.com/datasets). The model was trained using **Keras/TensorFlow** and is capable of recognizing eye states (open or closed) to monitor the driver's alertness.

Due to GitHub's file size limitations, the trained model file (`best_model.h5`) is not included in the repository.

---

## 🔗 Download the Model

To run this project, you need the trained model file (`best_model.h5`). You can download it from Google Drive:

📥 [Download the trained model here](https://drive.google.com/file/d/1mPbdeVGKRlMhYuArUsoCv2q2hLELuSpp/view?usp=sharing)

After downloading, place the model file in the `Models/` folder of this repository.

---

## 📦 Requirements

To run the project, you need to install the required dependencies:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
