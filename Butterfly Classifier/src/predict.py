import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load model and label encoder
model = load_model("models/best_model.keras")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load test CSV
test_df = pd.read_csv("data/Testing_set.csv")

# Image generator for test data
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    directory="data/test",
    x_col="filename",
    y_col=None,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# Predict
preds = model.predict(test_generator)
predicted_indices = np.argmax(preds, axis=1)
predicted_labels = le.inverse_transform(predicted_indices)

# Create submission
submission = pd.DataFrame({
    "filename": test_df["filename"],
    "label": predicted_labels
})
submission.to_csv("outputs/submission.csv", index=False)
print("Submission file saved to outputs/submission.csv")
