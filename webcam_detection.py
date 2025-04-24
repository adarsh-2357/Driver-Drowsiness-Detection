
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
from twilio.rest import Client
import time

# Initialize the Pygame mixer for the alarm
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')  # Replace with actual path if needed

# Load pre-trained DNN face detection model
face_net = cv2.dnn.readNetFromCaffe('Models/deploy.prototxt', 'Models/res10_300x300_ssd_iter_140000.caffemodel')

# Load the eye state detection model
model = tf.keras.models.load_model('Models/best_model.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Alarm and call parameters
alarm_threshold = 15
alarm_duration = 5
Score = 0
alarm_triggered = False
alarm_start_time = None

account_sid = 'your_twilio_sid_here'
auth_token = 'your_twilio_auth_token_here'
twilio_number = '+1234567890'
user_number = '+919999999999'
client = Client(account_sid, auth_token)
call_triggered = False

def make_call():
    try:
        ngrok_url = "https://your-ngrok-url.ngrok-free.app/voice.xml"
        call = client.calls.create(
            to=user_number,
            from_=twilio_number,
            url=ngrok_url
        )
        print(f"Call initiated: {call.sid}")
    except Exception as e:
        print(f"Error making the call: {e}")

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    face_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, w, h) = box.astype("int")

            face_width = w - x
            face_height = h - y
            if face_width > 100 and face_height > 100:
                face_detected = True
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

                face_roi = frame[y:h, x:w]
                eye = cv2.resize(face_roi, (80, 80))
                eye = eye / 255.0
                eye = eye.reshape(1, 80, 80, 3)

                prediction = model.predict(eye)

                if prediction[0][0] > 0.30:
                    Score += 1
                    cv2.putText(frame, 'closed', (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                    if Score >= alarm_threshold and not alarm_triggered:
                        alarm_start_time = cv2.getTickCount()
                        alarm_triggered = True
                        alarm_sound.play()

                elif prediction[0][1] > 0.75:
                    Score = max(0, Score - 2)
                    cv2.putText(frame, 'open', (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                    if alarm_triggered and (cv2.getTickCount() - alarm_start_time) / cv2.getTickFrequency() >= alarm_duration:
                        alarm_sound.stop()
                        alarm_triggered = False

    if Score >= 100 and not call_triggered:
        print("Score reached 100. Triggering the call...")
        make_call()
        call_triggered = True

    if Score < 100 and call_triggered:
        call_triggered = False

    if not face_detected:
        cv2.putText(frame, 'No Face Detected', (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    cv2.putText(frame, 'Score: ' + str(Score), (100, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
