import os
import sys
import cv2
import requests
import pyttsx3
import joblib
import time
import sklearn


# Define base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_RECOGNITION_DIR = os.path.join(BASE_DIR, "Gesture-Recognition")

ASL_TRAIN_DIR = os.path.join(BASE_DIR, "asl_alphabet_train")

# Add Gesture-Recognition to Python path for module imports
sys.path.insert(0, GESTURE_RECOGNITION_DIR)

# Import model utilities
from src.hand_tracker_nms import HandTrackerNMS
import src.extra

# Define model paths inside Gesture-Recognition
PALM_MODEL_PATH = os.path.join(GESTURE_RECOGNITION_DIR, "models", "palm_detection_without_custom_op.tflite")
LANDMARK_MODEL_PATH = os.path.join(GESTURE_RECOGNITION_DIR, "models", "hand_landmark.tflite")
ANCHORS_PATH = os.path.join(GESTURE_RECOGNITION_DIR, "models", "anchors.csv")
GESTURE_CLF_PATH = os.path.join(GESTURE_RECOGNITION_DIR, "models", "gesture_clf.pkl")

# Load gesture classification model
gesture_clf = joblib.load(GESTURE_CLF_PATH)

if not hasattr(gesture_clf, 'var_'):
    print("Warning: Gesture classifier model may not be trained or compatible.")


# Hand detection setup
detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

# Class mapping
int_to_char = src.extra.classes
connections = src.extra.connections

engine = pyttsx3.init()

def speak_text(text):

    engine.say(text)
    engine.runAndWait()


# Simulated ASL phrase list
phrases = ["HELLO", "HOWAREYOU", "GOODMORNING"]

def predict_gesture(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bboxes, joints = detector(image_rgb)
    if points is not None:
        pred_sign = src.extra.predict_sign(joints, gesture_clf, int_to_char)
        return pred_sign, points
    else:
        return "", None

print("Loaded model type:", type(gesture_clf))

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting webcam. Press 't' to translate, 'q' to quit.")

    buffer = []
    letter = ""
    static_gesture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_sign, points = predict_gesture(frame)
        if points is not None:
            src.extra.draw_points(points, frame)


        if pred_sign == letter:
            static_gesture_count += 1
        else:
            letter = pred_sign
            static_gesture_count = 0

       # Add letter after it stays the same for 6 frames
        if static_gesture_count > 6 and letter != "":
            buffer.append(letter)
            static_gesture_count = 0



        # If no hand detected (letter == ""), count static_gesture_count to add space
        if letter == "":
            static_gesture_count += 1
            if static_gesture_count > 12 and (len(buffer) == 0 or buffer[-1] != " "):
                buffer.append(" ")
                static_gesture_count = 0

        # Limit buffer size
        if len(buffer) > 6:
            buffer.pop(0)

        current_seq = ''.join(buffer)


        # Display buffer and last letter
        cv2.putText(frame, f"Buffer: {current_seq}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Last Letter: {letter}", (10, 80),
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

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()