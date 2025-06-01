import cv2
import numpy as np
import requests
import pyttsx3
import time

# Simulated ASL phrase list
phrases = ["HELLO", "HOWAREYOU", "GOODMORNING"]
phrase_index = 0
letter_index = 0

def predict_gesture_simulated():
    global phrase_index, letter_index
    # time.sleep(1.5)  Simulate signing delay

    if np.random.rand() < 0.5:
        return ""  # no gesture detected

    current_phrase = phrases[phrase_index]
    gesture = current_phrase[letter_index]
    letter_index += 1

    if letter_index >= len(current_phrase):
        letter_index = 0
        phrase_index = (phrase_index + 1) % len(phrases)

    return gesture



def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Start webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Starting webcam. Press 't' to translate, 'q' to quit.")
buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate ASL gesture detection
    gesture = predict_gesture_simulated()

    if gesture != "":
        buffer.append(gesture)
        if len(buffer) > 10:
            buffer.pop(0)

    current_seq = ''.join(buffer)

    # Display buffer and simulated letter
    cv2.putText(frame, f"Buffer: {current_seq}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Simulated Letter: {gesture}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("ASL Camera", frame)

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
