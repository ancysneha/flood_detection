import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC

# Paths
DATASET_PATH = "dataset/"
CATEGORIES = ["flooded", "non_flooded"]

# HOG feature parameters
hog_params = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}

features = []
labels = []

print("\nLoading images and extracting features...\n")

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_PATH, category)
    
    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)
        
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (128, 128))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract HOG features
            hog_feature = hog(gray, **hog_params)
            
            features.append(hog_feature)
            labels.append(label)
        
        except:
            print(f"Skipping corrupted file: {path}")

# Convert to numpy array
features = np.array(features)
labels = np.array(labels)

print("\nTraining SVM model...\n")

# Train SVM
model = LinearSVC()
model.fit(features, labels)

# Save model
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
