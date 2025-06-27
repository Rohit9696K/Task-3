# Task-3
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Set the path to the dataset (adjust this)
DATA_DIR = "path_to_your_dataset/train"  # e.g., "/content/train"

# Image size (resize to smaller size for faster computation)
IMG_SIZE = 64

# Load images and labels
def load_data():
    images = []
    labels = []
    for file in os.listdir(DATA_DIR):
        label = 1 if "dog" in file else 0
        img_path = os.path.join(DATA_DIR, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use grayscale for simplicity
        images.append(img.flatten())  # Flatten 2D to 1D
        labels.append(label)

        if len(images) >= 2000:  # Load only 2000 samples for speed
            break
    return np.array(images), np.array(labels)

print("Loading data...")
X, y = load_data()
print("Data loaded:", X.shape)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
