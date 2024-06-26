import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

dir = r"./Data/Dataset_corretto"
   
image_size = (200, 200)
batch_size = 64
validation_split = 0.2
train_test_split_ratio = 0.2 

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split
)

dataset = datagen.flow_from_directory(
    dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=42,
    interpolation="bilinear"
)

df_val = datagen.flow_from_directory(
    dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=42,
    interpolation="bilinear"
)

filepaths = dataset.filepaths
labels = dataset.classes

# Split del set di addestramento per creare il set di test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    filepaths, labels, test_size=train_test_split_ratio, stratify=labels, random_state=42)

kernel_size = (3,3)

model = Sequential([
     Conv2D(filters=16, kernel_size=kernel_size, activation='relu', input_shape=(400, 400, 1)),
     MaxPooling2D(2,2),

     Conv2D(filters=32, kernel_size=kernel_size, activation='relu'),
     MaxPooling2D(2,2),

     Conv2D(filters=64, kernel_size=kernel_size, activation='relu'),
     MaxPooling2D(2,2),

     Conv2D(filters=128, kernel_size=kernel_size, activation='relu'),
     MaxPooling2D(2,2),

     Flatten(),

     Dense(200, activation='relu'),
     Dropout(0.3), 
     
     Dense(100, activation='relu'),
     Dropout(0.2), 
     
     Dense(50, activation='relu'),
     Dropout(0.1),
     Dense(25, activation='relu'),


     Dense(50, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epoche = 70

model.summary()

early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    
history = model.fit(dataset, epochs=epoche, validation_data=df_val, batch_size=batch_size,callbacks = early_stopping)

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


images = np.concatenate([batch[0].numpy() for batch in test_labels], axis=0)
labels = np.concatenate([batch[1].numpy() for batch in test_labels], axis=0)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

y_hat = model.predict(images)
print("Predictions:", y_hat)


for images, labels in test_paths.take(1):
    y_hat = model.predict(images)
    print("Predictions:\n", y_hat)
    print("True Labels:\n", labels)

class_names = dataset.class_names

plt.figure(figsize=(10, 10))

for images, labels in test_paths.take(1):   
    y_hat = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        plt.title(f"Pred: {class_names[np.argmax(y_hat[i])]} \n True: {class_names[np.argmax(labels[i])]}")
        plt.axis("off")
        
plt.show()

