# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Load malaria dataset from TensorFlow Datasets
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])

# Split dataset into train, validation, and test sets
def split(dataset, TRAIN_RATIO, TEST_RATIO, VAL_RATIO):
    dataset_size = len(dataset)
    train_dataset = dataset.take(int(dataset_size * TRAIN_RATIO))
    val_dataset = dataset.skip(int(dataset_size * TRAIN_RATIO)).take(int(dataset_size * VAL_RATIO))
    test_dataset = dataset.skip(int(dataset_size * (TRAIN_RATIO + VAL_RATIO)))
    return train_dataset, val_dataset, test_dataset

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

train_dataset, val_dataset, test_dataset = split(dataset[0], TRAIN_RATIO, TEST_RATIO, VAL_RATIO)

# Preprocess images: resize, rescale, normalize
IM_SIZE = 224

def resize_rescale(image, label):
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0
    image = tf.ensure_shape(image, (IM_SIZE, IM_SIZE, 3))
    return image, label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

# Define CNN model
model = tf.keras.models.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    Conv2D(filters=6, kernel_size=3, strides=1, padding='valid', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    Flatten(),
    Dense(1000, activation='relu'),
    BatchNormalization(),
    Dense(100, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset.batch(32),
                    validation_data=val_dataset.batch(32),
                    batch_size=32,
                    epochs=100,
                    verbose=1)

# Evaluate the model on test dataset
model.evaluate(test_dataset.batch(32))

# Define function to predict and display results
def parasite_or_not(x):
    if x < 0.5:
        return 'Parasitized'
    else:
        return 'Uninfected'

# Display sample predictions
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(test_dataset.take(16)):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image.numpy())
    plt.title(f'{parasite_or_not(model.predict(image[tf.newaxis, ...])[0][0])}: {label.numpy()[0]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
