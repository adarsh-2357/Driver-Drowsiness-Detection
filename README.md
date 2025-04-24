# Driver Drowsiness Detection

This project uses deep learning and computer vision techniques to detect driver drowsiness based on eye images. A pretrained InceptionV3 model is fine-tuned on the MRL Eye Dataset to classify whether the driver is drowsy or alert.

---

## 🧠 Model
- **Base Model:** InceptionV3 (pretrained on ImageNet)
- **Custom Layers:** Flatten → Dense → Dropout → Dense (Softmax)
- **Output:** 2 Classes – Drowsy, Alert

---

## 📁 Dataset
- **MRL Eye Dataset**
- Used for training, validation, and testing
- Loaded using `ImageDataGenerator` from Keras

---

## 🔧 How to Train

> Model training was performed on [Kaggle](https://www.kaggle.com) with the following:
- TensorFlow 2.13.0
- Keras 2.13.1
- ImageDataGenerator for augmentation
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks

---

## 💾 Model Saving

The best model is saved as:
- `best_model.h5` (HDF5 format)
- `best_model/` (TensorFlow SavedModel format)

---

## 📈 Results

- **Training Accuracy:** _(Insert after running the evaluation)_
- **Validation Accuracy:** _(Insert here)_
- **Test Accuracy:** _(Insert here)_

---

## 🚀 Future Improvements

- Integrate the model with a webcam to detect drowsiness in real-time
- Deploy on edge devices (like Raspberry Pi or Jetson Nano)
- Add sound alerts when drowsiness is detected

---

## 📌 Note

> This was a learning project created using Kaggle Notebooks and OpenCV-based dataset preprocessing. It helped reinforce concepts in transfer learning, image augmentation, and model evaluation.

---

## 📚 Acknowledgements

- MRL Eye Dataset: [MRL Dataset Paper / Source]([https://](https://universe.roboflow.com/mrl-eye-dataset-rwrm0))
- Kaggle: For GPU support and training environment

---

Feel free to contribute or raise issues if you'd like to expand the project!
