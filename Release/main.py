import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, CenterCrop
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image_dataset_from_directory
from tensorflow.data import AUTOTUNE
import os

if __name__ == '__main__':
    
    train_set_path = r"./Data/Train_set"
    test_set_path = r"./Data/Test_set"
    valid_set_path = r"./Data/Val_set"
    
    batch_size = 32
    train_set = image_dataset_from_directory(
        directory = train_set_path,
        labels='inferred', 
        label_mode='categorical',
        color_mode="grayscale",
        image_size=(400, 400),
        batch_size = batch_size,
        shuffle=True,
        seed=42,
        interpolation="bilinear"
    )
    
    test_set = image_dataset_from_directory(
        directory = test_set_path,
        labels='inferred',
        label_mode='categorical',
        color_mode="grayscale",
        image_size=(400, 400),
        batch_size = batch_size,
        shuffle=True,
        seed=42,
        interpolation="bilinear"
    )
    
    valid_set= image_dataset_from_directory(
        directory = valid_set_path,
        labels='inferred',
        label_mode='categorical',
        color_mode="grayscale",
        image_size=(400, 400),
        batch_size = batch_size,
        shuffle=True,
        seed=42,
        interpolation="bilinear",
    )
    

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
        Dropout(0.2),

        Dense(100, activation='relu'),
        Dropout(0.2), 

        Dense(50, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    epoche = 40

    model.summary()

    early_stopping = EarlyStopping(
            monitor='accuracy',
            patience=3,
            restore_best_weights=True
    )
    
    AUTOTUNE = AUTOTUNE
    
    train_ds = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_set.cache().prefetch(buffer_size=AUTOTUNE) 
        
    history = model.fit(train_ds, epochs=epoche, validation_data=valid_ds, batch_size=batch_size,callbacks = early_stopping)

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


    model.save(filepath=r'./model\lego.keras')

    #model = load_model(r'./model/lego.keras')
    
    filepaths = test_set.take(1)
    print("Total batches seen:", filepaths )

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

    class_names = train_set.class_names

    plt.figure(figsize=(10, 10))

    for images, labels in test_set.take(1):   
        y_hat = model.predict(images)
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(f"Pred: {class_names[np.argmax(y_hat[i])]} \n True: {class_names[np.argmax(labels[i])]}")
            plt.axis("off")
            
    plt.show()

