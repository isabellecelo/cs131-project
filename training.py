# train_classifier.py

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


# Load preprocessed data
data = np.load('gesture_dataset.npz')

X = data['data']  # features (e.g., distances)
y = data['labels']  # labels (e.g., letter classes)

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation accuracy: {accuracy:.2f}")

scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-val accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# Save the trained model
joblib.dump(clf, 'Gesture-Recognition/models/gesture_clf.pkl')
