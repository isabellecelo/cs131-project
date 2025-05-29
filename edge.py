import cv2
import numpy as np
import tensorflow as tf
import requests
import pyttsx3
import random
import time

# Load your model (replace with actual model if you have one)
# model = tf.keras.models.load_model("asl_alphabet_model.h5")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def predict_gesture_simulated():
    # Simulate gesture every 2 seconds (placeholder for model)
    time.sleep(2)
    return random.choice(labels)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
print("Starting webcam. Press 'q' to quit.")
buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate gesture prediction
    gesture = predict_gesture_simulated()
    buffer.append(gesture)

    # Limit buffer size
    if len(buffer) > 10:
        buffer.pop(0)

    current_seq = ''.join(buffer)

    # Show webcam and detected letters
    cv2.putText(frame, f"Buffer: {current_seq}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Camera", frame)

    # Press 't' to trigger translation
    key = cv2.waitKey(1)
    if key == ord('t'):
        print("Sending:", current_seq)
        try:
            response = requests.post("http://127.0.0.1:5000/translate", json={"input": current_seq})
            translation = response.json().get("translation", "[no response]")
            print("Translated:", translation)
            speak_text(translation)
        except Exception as e:
            print("Error contacting server:", e)

        buffer = []

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
