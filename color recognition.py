#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths to your dataset
dataset_dir = 'C:/Users/wt/Desktop/dataset_colors'
train_dir = 'path/to/train_dataset'
test_dir = 'path/to/test_dataset'

# Create directories for training and testing datasets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# In[2]:


# Get a list of all the classes (subfolder names)
classes = os.listdir(dataset_dir)

# Split ratio (e.g., 80% for training, 20% for testing)
split_ratio = 0.8

# Iterate over each class to split its images
for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    images = os.listdir(class_dir)

    # Split the images into training and testing sets
    train_images, test_images = train_test_split(images, train_size=split_ratio, random_state=42)

    # Create class-specific subdirectories in training and testing directories
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Move the images to the respective directories
    for image in train_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(train_dir, cls))

    for image in test_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(test_dir, cls))

print("Dataset successfully split into training and testing sets.")


# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data generator for preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Load and preprocess testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


# In[5]:


from tensorflow.keras import models, layers

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()


# In[6]:


# Train the CNN model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator
)


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Generate predictions and confusion matrix
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print classification report
print('Classification Report')
target_names = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred, target_names=target_names))


# In[13]:


# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()


# In[ ]:


# Save the trained model
model.save('color_classification_cnn.h5')

