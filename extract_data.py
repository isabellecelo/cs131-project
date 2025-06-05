import cv2
import numpy as np
from tqdm import tqdm
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_RECOGNITION_DIR = os.path.join(BASE_DIR, "Gesture-Recognition")
sys.path.insert(0, GESTURE_RECOGNITION_DIR)


from src.hand_tracker_nms import HandTrackerNMS
import src.extra

MAX_DIM = 320

# Paths
DATASET_PATH = "asl_alphabet_train"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


detector = HandTrackerNMS(
    "Gesture-Recognition/models/palm_detection_without_custom_op.tflite",
    "Gesture-Recognition/models/hand_landmark.tflite",
    "Gesture-Recognition/models/anchors.csv"
)

data = []
labels = []

for label in tqdm(LABELS):
    folder = os.path.join(DATASET_PATH, label)
    for img_file in os.listdir(folder)[:200]:
        img_path = os.path.join(folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points, bboxes, joints = detector(image_rgb)
        if joints is not None:
            distances = src.extra.calc_distances(joints)  
            data.append(distances)
            labels.append(label)

np.savez("gesture_dataset.npz", data=np.array(data), labels=np.array(labels))
