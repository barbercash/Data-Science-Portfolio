# src/train.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
train_df = pd.read_csv("data/Training_set.csv")
train_df["label"] = train_df["label"].astype(str)

# Encode labels
le = LabelEncoder()
train_df["encoded"] = le.fit_transform(train_df["label"])

# Split
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=42)

# Image generators
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

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(75, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("models/butterfly_model.h5", save_best_only=True)
]

model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=callbacks)

# Save label encoder
import pickle
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
