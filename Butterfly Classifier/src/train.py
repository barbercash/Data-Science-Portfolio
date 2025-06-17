# src/train.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Load data
train_df = pd.read_csv("data/Training_set.csv")
train_df["label"] = train_df["label"].astype(str)

# Encode labels
le = LabelEncoder()
train_df["encoded"] = le.fit_transform(train_df["label"])

# Split into train and validation
train_data, val_data = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=42
)

# Image data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_dataframe(
    train_data,
    directory="data/train",
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_gen.flow_from_dataframe(
    val_data,
    directory="data/train",
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(75, activation='softmax')  # 75 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and best model saving
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("models/best_model.keras", save_best_only=True)
]

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# Save final model (last epoch)
model.save("models/final_model.keras")

# Save the label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
