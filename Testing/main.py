import matplotlib.pyplot as plt
import sklearn  
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer
from keras import preprocessing as pre
from keras import Sequential
from keras import layers as lay
from keras import layers, metrics, optimizers, models, losses, utils
import numpy as np  
import tensorflow as tf


dir = r"C:\Users\BrunoLuciano\OneDrive - ITS Angelo Rizzoli\Documents\Deep_learning_project\Data\LEGO brick images v1"

df_train, df_val = pre.image_dataset_from_directory(
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

model = Sequential([
    lay.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(200, 200, 1)),
    lay.MaxPooling2D(2,2),

    lay.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    lay.MaxPooling2D(2,2),

    lay.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    lay.MaxPooling2D(2,2),

    lay.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    lay.MaxPooling2D(2,2),

    lay.Flatten(),

    lay.Dense(128, activation='relu'),
    lay.Dropout(0.3),  

    lay.Dense(64, activation='relu'),
    lay.Dropout(0.2),

    lay.Dense(32, activation='relu'),
    lay.Dense(16, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epoche = 40

model.summary()
    
history = model.fit(df_train, epochs=epoche, validation_data=df_val, batch_size=32)

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


model.save(filepath=r'C:\Users\BrunoLuciano\OneDrive - ITS Angelo Rizzoli\Documents\Deep_learning_project\model\lego.h5')


for images, labels in df_val:
    predictions = model.predict(images)
    print("Predictions:\n", predictions)
    print("True Labels:\n", labels.numpy())

class_names = df_train.class_names

plt.figure(figsize=(10, 10))
for images, labels in df_val:   
    predictions = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        plt.title(f"Pred: {class_names[np.argmax(predictions[i])]} \n True: {class_names[np.argmax(labels[i])]}")
        plt.axis("off")
plt.show()
