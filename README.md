# Driver Drowsiness Detection

## Project Description
This project implements a robust real-time driver drowsiness detection system. It leverages deep learning and computer vision techniques to monitor a driver's eye state via a live webcam feed. Upon detecting signs of drowsiness for a prolonged period, the system triggers an audible alarm. For more critical situations, it's also capable of initiating an automated phone call alert using the Twilio API, significantly enhancing road safety.

## Features
* **Real-time Monitoring:** Continuously analyzes the driver's face and eyes from a live webcam feed.
* **Advanced Face Detection:** Utilizes a pre-trained **DNN (Deep Neural Network)** model (`Caffe` framework) for accurate and efficient face detection.
* **Deep Learning Eye Classifier:** Employs a custom-trained **Convolutional Neural Network (CNN)** based on the **InceptionV3** architecture to classify eye states (open/closed) directly from detected eye regions.
* **Audible Alarm System:** Triggers an alarm sound (`alarm.wav`) when drowsiness is detected for a configurable duration (`alarm_threshold = 15` frames).
* **Automated Phone Call Alerts:** Integrates with the **Twilio API** to make an automated phone call to a predefined number if drowsiness persists beyond a higher threshold (`Score >= 100`).
* **Robust Model:** The eye classification model was fine-tuned using the comprehensive **MRL Eye Dataset**, ensuring strong performance across various conditions.

---

## How It Works

The system operates through a series of sequential steps to ensure accurate and timely drowsiness detection:

1.  **Webcam Initialization:** The system initializes the default webcam (`cv2.VideoCapture(0)`) to capture real-time video frames.
2.  **Face Detection:** For each captured frame, a pre-trained Caffe-based DNN model (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) is used to detect faces. It identifies faces with a confidence greater than 50%.
3.  **Eye Region Extraction & Preprocessing:** If a sufficiently large face (width and height > 100 pixels) is detected, the region of interest (ROI) containing the face is extracted. This ROI is then resized to `80x80` pixels and normalized (`/ 255.0`) to match the input requirements of the eye classification model.
4.  **Eye State Classification:** The preprocessed eye region is fed into the loaded **InceptionV3-based Keras model** (`best_model.h5`). The model outputs a prediction array.
    * If `prediction[0][0]` (likely corresponding to 'closed' eye probability) is greater than `0.30`, it indicates closed eyes.
    * If `prediction[0][1]` (likely corresponding to 'open' eye probability) is greater than `0.75`, it indicates open eyes.
5.  **Drowsiness Scoring:** A `Score` variable tracks the level of drowsiness.
    * It **increments by 1** when eyes are classified as 'closed'.
    * It **decrements by 2** (down to a minimum of 0) when eyes are classified as 'open'.
6.  **Alarm Triggering:**
    * If the `Score` reaches or exceeds `alarm_threshold` (set to `15`), an audible alarm (`alarm.wav`) starts playing. The alarm will continue playing as long as `Score` remains above the threshold for `alarm_duration` (set to `5` seconds) and will stop if the driver's eyes open consistently.
7.  **Automated Phone Call Alert:**
    * If the `Score` reaches or exceeds `100`, the system initiates an automated phone call via the configured Twilio API to a predefined `user_number`. This call sends a voice message (retrieved from an `ngrok_url`). The call is triggered only once per drowsiness episode (controlled by `call_triggered` flag).
8.  **Visual Feedback:** The system continuously displays the webcam feed with detected faces, current eye status ('open'/'closed'), and the `Score` value.

---

## Technologies Used
* **Programming Language:** Python
* **Deep Learning Frameworks:** Keras, TensorFlow
* **Computer Vision Libraries:** OpenCV, Dlib (for potential underlying facial landmark operations, though direct DNN face detection is used in `webcam_detection.py`), imutils
* **Audio Playback:** Pygame Mixer
* **API Integration:** Twilio (for automated phone calls)

---

## Setup and Installation

To get this project running on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/adarsh-2357/Driver-Drowsiness-Detection.git](https://github.com/adarsh-2357/Driver-Drowsiness-Detection.git)
    cd Driver-Drowsiness-Detection
    ```

2.  **Download Required Models:**
    Due to file size limitations on GitHub, the trained deep learning model (`best_model.h5`) and the DNN face detection models are not included directly in the repository.

    * **Eye State Detection Model (`best_model.h5`):**
        Please download it from the following Google Drive link:
        **[Download best_model.h5 from Google Drive](https://drive.google.com/file/d/10G8n4S8zV4R-u1r2z4M_w8cW8qJ5T1w/view?usp=sharing)**

    * **DNN Face Detection Models:**
        You will need two files for the pre-trained Caffe model:
        * `deploy.prototxt`
        * `res10_300x300_ssd_iter_140000.caffemodel`
        These can typically be found in OpenCV's extra modules or by searching for "OpenCV DNN face detector models". If you don't find them, a quick search on GitHub or Google for "deploy.prototxt res10_300x300_ssd_iter_140000.caffemodel" will yield download links.

3.  **Place the Model Files:**
    After downloading, move all model files (`best_model.h5`, `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) into the `Models/` directory within your cloned repository. The directory structure should look like this:
    ```
    Driver-Drowsiness-Detection/
    ├── Models/
    │   ├── best_model.h5
    │   ├── deploy.prototxt
    │   └── res10_300x300_ssd_iter_140000.caffemodel
    ├── alarm.wav
    ├── driver_drowsiness_detection.ipynb
    └── webcam_detection.py
    ├── requirements.txt (Highly Recommended)
    └── ... (other project files)
    ```

4.  **Install Dependencies:**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    # Create a virtual environment (if you don't have one)
    python -m venv venv
    # Activate the virtual environment
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```
    *If you do not have a `requirements.txt` file, you'll need to install the following key libraries individually:*
    ```bash
    pip install tensorflow keras opencv-python dlib scipy imutils twilio pygame
    ```

---

## Twilio Integration (Optional)

To enable the automated phone call feature, you'll need to configure your Twilio account.

1.  **Sign up for Twilio:** Create an account at [Twilio.com](https://www.twilio.com/).
2.  **Get your Account SID and Auth Token:** Find these on your Twilio Console dashboard.
3.  **Get a Twilio Phone Number:** Purchase a Twilio phone number (it must have voice capabilities).
4.  **Expose Local Server with ngrok:** Twilio needs a publicly accessible URL to fetch the TwiML (XML instructions for the call). In your `webcam_detection.py` code, the `ngrok_url` is specified as `"https://your-ngrok-url.ngrok-free.app/voice.xml"`. You'll need to run `ngrok` to expose your local server where `voice.xml` (containing the TwiML) would reside.
    * Download ngrok from [ngrok.com](https://ngrok.com/download).
    * Run `ngrok http <your_local_port>` (e.g., if you have a simple web server serving `voice.xml` on port 8000).
    * Update the `ngrok_url` in your `webcam_detection.py` with the actual public URL provided by ngrok.
5.  **Update Credentials in `webcam_detection.py`:**
    Locate the following lines in your `webcam_detection.py` file and replace the placeholder values with your actual Twilio credentials and phone numbers:
    ```python
    account_sid = 'your_twilio_sid_here'      # Replace with your Twilio Account SID
    auth_token = 'your_twilio_auth_token_here' # Replace with your Twilio Auth Token
    twilio_number = '+1234567890'             # Replace with your Twilio Phone Number (e.g., '+15017122661')
    user_number = '+919999999999'             # Replace with the recipient's phone number (e.g., your own phone)
    ngrok_url = "[https://your-ngrok-url.ngrok-free.app/voice.xml](https://your-ngrok-url.ngrok-free.app/voice.xml)" # Replace with your actual ngrok URL
    ```
    *It is highly recommended to use environment variables for sensitive credentials in a production environment.*

---

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
    * **Webcam Feed:** A new window titled "Drowsiness Detection" will pop up, displaying your live webcam feed at `640x480` resolution.
    * **Face Detection Overlay:** The system will draw a blue rectangular bounding box around detected faces.
    * **Eye State Indicator:** On the bottom left of the window, text will appear indicating your eye status ('open' or 'closed').
    * **Drowsiness Score:** To the right of the eye status, a 'Score' will be displayed. This score increases when your eyes are closed and decreases when they are open.
    * **Audible Alarm:** If your 'Score' reaches or exceeds `15`, an audible alarm sound (`alarm.wav`) will begin to play. The alarm will stop if your eyes remain open consistently, causing the score to drop.
    * **Automated Phone Call (if configured):** If your 'Score' further increases to `100` and Twilio is correctly configured, the system will initiate an automated phone call to the `user_number`.
    * **No Face Detected:** If no face is detected in the frame, a "No Face Detected" message will be displayed in red.

* **Exiting the Application:**
    To stop the real-time detection and close the webcam feed, simply press the `'q'` key on your keyboard while the "Drowsiness Detection" window is active.

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
        * **Directory Setup & Imports:** Initial setup for Python libraries.
        * **Image Preprocessing:** Details on using `ImageDataGenerator` for data augmentation (rescaling, rotation, shear, zoom, shifts) and splitting data. Note `IMG_SIZE = 80` and `BATCH_SIZE = 8`.
        * **Model Architecture (InceptionV3):** Explanation of using `InceptionV3` as the base model (pre-trained on ImageNet, `include_top=False`, `trainable=False` for base layers). Custom layers include `Flatten`, `Dense(64, activation='relu')`, `Dropout(0.5)`, and the final `Dense(2, activation='softmax')` output layer.
        * **Compilation and Training:** Details on using `adam` optimizer, `categorical_crossentropy` loss, and `accuracy` metric. The training process uses `ModelCheckpoint` to save the best model, `EarlyStopping` for preventing overfitting, and `ReduceLROnPlateau` for learning rate adjustment over `30` epochs.
        * **Evaluation:** Scripts to evaluate the model's performance on both training and validation sets, printing accuracy and loss.
    * **Interactive Exploration:** You can modify code cells, run experiments, and observe the results directly within the notebook environment.

---

## Model Training Details

This section summarizes the key aspects of the deep learning model training process, as detailed in `driver_drowsiness_detection.ipynb`.

* **Base Model:** InceptionV3 (pre-trained on ImageNet)
* **Input Image Size:** 80x80 pixels with 3 color channels (`(80, 80, 3)`)
* **Batch Size:** 8
* **Data Augmentation:**
    * Rescaling pixel values to `[0, 1]`
    * Random rotation (up to 20 degrees)
    * Random shear (up to 20%)
    * Random zoom (up to 20%)
    * Random width and height shifts (up to 20%)
* **Custom Layers (on top of InceptionV3):**
    * `Flatten` layer
    * `Dense` layer with 64 units and `relu` activation
    * `Dropout` layer with 50% dropout rate
    * `Dense` output layer with 2 units (for 'open'/'closed' classes) and `softmax` activation
* **Optimizer:** Adam
* **Loss Function:** Categorical Cross-entropy (`categorical_crossentropy`)
* **Metrics:** Accuracy (`accuracy`)
* **Callbacks:**
    * `ModelCheckpoint`: Saves the best model based on validation accuracy (`best_model.h5`).
    * `EarlyStopping`: Stops training if validation loss doesn't improve for 7 epochs, restoring the best weights.
    * `ReduceLROnPlateau`: Reduces learning rate if validation loss plateaus for 3 epochs.
* **Epochs:** 30 (though early stopping might conclude training sooner)
* **Dataset:** MRL Eye Dataset

---

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

---

## Contact

For any questions or inquiries, please reach out to:
Adarsh Kumar
[adarsh20103@gmail.com]
