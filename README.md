# Driver Drowsiness Detection

## Project Description
This project implements a real-time driver drowsiness detection system using deep learning. It analyzes live webcam feed to monitor the driver's eye state and triggers an alarm if signs of drowsiness are detected. For prolonged periods of drowsiness, the system can also initiate an automated phone call alert.

## Features
* **Real-time Detection:** Monitors driver's face and eyes in real-time using a webcam.
* **Deep Learning Model:** Utilizes a custom-trained neural network based on the InceptionV3 architecture for accurate eye state classification (open/closed).
* **Audible Alarm:** Triggers an alarm sound if drowsiness is detected for a specified duration.
* **Automated Phone Call Alert:** Integrates with the Twilio API to make an automated phone call for extended drowsiness events.
* **Custom Dataset:** Model fine-tuned using the MRL Eye Dataset.

## Technologies Used
* **Programming Languages:** Python
* **Deep Learning Frameworks:** Keras, TensorFlow
* **Libraries:** OpenCV, Dlib (likely used for facial landmark detection)
* **API:** Twilio

## Setup and Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/adarsh-2357/Driver-Drowsiness-Detection.git](https://github.com/adarsh-2357/Driver-Drowsiness-Detection.git)
    cd Driver-Drowsiness-Detection
    ```

2.  **Download the Trained Model:**
    Due to file size limitations on GitHub, the trained model (`best_model.h5`) is not included in the repository. Please download it from the following Google Drive link:
    [Google Drive Link to best\_model.h5](https://drive.google.com/file/d/10G8n4S8zV4R-u1r2z4M_w8cW8qJ5T1w/view?usp=sharing) (Note: Please replace with the actual working link if it's different or broken)

3.  **Place the Model File:**
    After downloading, move the `best_model.h5` file into the `Models/` directory within your cloned repository.
    ```
    Driver-Drowsiness-Detection/
    ├── Models/
    │   └── best_model.h5
    ├── alarm.wav
    ├── driver_drowsiness_detection.ipynb
    └── webcam_detection.py
    ```

4.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt # (Assuming you have a requirements.txt file, otherwise list individual dependencies)
    ```
    *If you don't have a `requirements.txt` file, you'll need to install the following key libraries:*
    ```bash
    pip install tensorflow keras opencv-python dlib scipy imutils twilio
    ```

## Usage

This project provides two primary ways to interact with the system: running the real-time detection script and exploring the model's development in a Jupyter Notebook.

### 1. Running the Real-time Driver Drowsiness Detection

This is the main application of the project, enabling live drowsiness detection using your webcam.

* **Execution Command:**
    Open your terminal or command prompt, navigate to the `Driver-Drowsiness-Detection` directory, and run:
    ```bash
    python webcam_detection.py
    ```

* **What to Expect:**
    * **Webcam Feed:** A new window will pop up displaying your live webcam feed.
    * **Face and Eye Detection:** The system will attempt to detect your face and eyes in real-time. You should see bounding boxes or landmarks drawn around your face and eyes.
    * **Drowsiness Status:** On the video feed, you will likely see a visual indicator (e.g., text like "Drowsy!" or "Awake") displaying your current drowsiness status.
    * **Audible Alarm:** If your eyes are detected as closed for a consecutive number of frames (indicating drowsiness), an audible alarm sound (`alarm.wav`) will play to alert you.
    * **Automated Phone Call (if configured):** For prolonged or severe drowsiness, if you have configured the Twilio API credentials, the system will attempt to make an automated phone call to the predefined recipient.
    * **Performance:** The performance (frame rate, detection speed) may vary depending on your computer's processing power and webcam quality.

* **Exiting the Application:**
    To stop the real-time detection and close the webcam feed, simply press the `'q'` key on your keyboard while the detection window is active.

### 2. Exploring the Model Training and Evaluation (Jupyter Notebook)

The `driver_drowsiness_detection.ipynb` notebook provides a comprehensive walkthrough of how the deep learning model was developed, trained, and evaluated. This is invaluable if you want to understand the underlying AI, retrain the model, or experiment with different architectures/datasets.

* **Execution Command:**
    From your terminal, in the `Driver-Drowsiness-Detection` directory, run:
    ```bash
    jupyter notebook driver_drowsiness_detection.ipynb
    ```

* **What to Expect:**
    * **Browser Interface:** This command will open a new tab in your web browser displaying the Jupyter Notebook interface.
    * **Code Cells:** The notebook is composed of cells containing Python code, markdown text, and outputs. You can execute cells individually or run all cells.
    * **Detailed Walkthrough:**
        * **Data Loading & Preprocessing:** See how the MRL Eye Dataset was loaded, augmented, and prepared for training.
        * **Model Architecture:** Examine the InceptionV3-based Convolutional Neural Network (CNN) architecture used for eye classification.
        * **Training Process:** Visualize the training progress, including loss and accuracy curves over epochs.
        * **Model Evaluation:** Understand how the model's performance was evaluated using metrics like accuracy, precision, recall, and F1-score on a test set.
        * **Model Saving:** See the code used to save the `best_model.h5` file.
    * **Interactive Exploration:** You can modify code cells, run experiments, and observe the results directly within the notebook environment.

## Project Structure
* `Models/`: Contains the trained deep learning model.
* `alarm.wav`: The audio file used for the drowsiness alarm.
* `driver_drowsiness_detection.ipynb`: Jupyter Notebook containing model training, evaluation, and possibly initial setup.
* `webcam_detection.py`: The main script for real-time drowsiness detection using the webcam.
