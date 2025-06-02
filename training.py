# train_classifier.py

import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load preprocessed data
data = np.load('gesture_dataset.npz')

X = data['data']  # features (e.g., distances)
y = data['labels']  # labels (e.g., letter classes)

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(clf, 'Gesture-Recognition/models/gesture_clf.pkl')
