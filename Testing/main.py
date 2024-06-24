import matplotlib.pyplot as plt
from keras.preprocessing import image_dataset_from_directory
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

dir = r".\Data\LEGO brick images v1"


dataset, df_val = image_dataset_from_directory(
    directory = dir,
    labels='inferred', 
    label_mode='categorical',
    color_mode="grayscale",
    image_size=(200, 200),
    batch_size = 128,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    validation_split=0.2,
    subset='both'
)

dataset_size = len(dataset)
print(f"Dataset size: {dataset_size}")

train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
print(f"Train size: {train_size}, Test size: {test_size}")

train_set = dataset.take(train_size)

test_set = dataset.skip(test_size)



model = Sequential([
     Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(200, 200, 1)),
     MaxPooling2D(2,2),

     Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
     MaxPooling2D(2,2),

     Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
     MaxPooling2D(2,2),

     Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
     MaxPooling2D(2,2),

     Flatten(),

     Dense(128, activation='relu'),
     Dropout(0.3),  

     Dense(64, activation='relu'),
     Dropout(0.2),

     Dense(32, activation='relu'),
     Dropout(0.1),

     Dense(16, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epoche = 50

model.summary()

early_stopping = EarlyStopping(
        monitor='accuracy',
        patience=5,
        restore_best_weights=True
    )

    
history = model.fit(dataset, epochs=epoche, validation_data=df_val, batch_size=32,callbacks = early_stopping)

# Accuratezza
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.save(filepath=r'.\model\lego.keras')


images = np.concatenate([batch[0].numpy() for batch in test_set], axis=0)
labels = np.concatenate([batch[1].numpy() for batch in test_set], axis=0)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

y_hat = model.predict(images)
print("Predictions:", y_hat)


for images, labels in test_set.take(1):
    y_hat = model.predict(images)
    print("Predictions:\n", y_hat)
    print("True Labels:\n", labels)

class_names = dataset.class_names

plt.figure(figsize=(10, 10))

for images, labels in test_set.take(1):   
    y_hat = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        plt.title(f"Pred: {class_names[np.argmax(y_hat[i])]} \n True: {class_names[np.argmax(labels[i])]}")
        plt.axis("off")
        
plt.show()

