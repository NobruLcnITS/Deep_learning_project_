import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,CenterCrop
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os 

if __name__ == '__main__':
        
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dir = r"./Data/Dataset_corretto"
    train_set_path = r"./Data/Train_set"
    test_set_path = r"./Data/Test_set"
    valid_set_path = r"./Data/Val_set"
    
    image_size = (300, 300)
    batch_size = 32
    validation_split = 0.2
    trainin_split = 0.8
    train_test_split_ratio = 0.2 

    datagen_1 = ImageDataGenerator(
        rescale=1./255,
        
    )
    
    
    train_set = datagen_1.flow_from_directory(
        directory=train_set_path,
        class_mode='categorical',
        color_mode="grayscale",
        target_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        interpolation="bilinear",

    )
    
    
    valid_set= datagen_1.flow_from_directory(
        directory = valid_set_path,
        class_mode='categorical',
        color_mode="grayscale",
        target_size=image_size,
        batch_size = batch_size,
        shuffle=True,
        seed=42,
        interpolation="bilinear",
       
    )
    
    datagen_2= ImageDataGenerator(
        rescale=1./255,
        
    )
    
    test_set = datagen_2.flow_from_directory(
        directory = test_set_path,
        class_mode='categorical',
        color_mode="grayscale",
        target_size=image_size,
        batch_size = batch_size,
        shuffle=True,
        seed=42,
        interpolation="bilinear"
    )
    
    
    # kernel_size = (3,3)

    # model = Sequential([
    #     Conv2D(filters=16, kernel_size=kernel_size, activation='relu', input_shape=(300, 300, 1)),
    #     MaxPooling2D(2,2),
        
    #     Conv2D(filters=32, kernel_size=kernel_size, activation='relu'),
    #     MaxPooling2D(2,2),

    #     Conv2D(filters=64, kernel_size=kernel_size, activation='relu'),
    #     MaxPooling2D(2,2),

    #     Conv2D(filters=128, kernel_size=kernel_size, activation='relu'),
    #     MaxPooling2D(2,2),

    #     Flatten(),

    #     Dense(200, activation='relu'),
    #     Dropout(0.2),

    #     Dense(100, activation='relu'),
    #     Dropout(0.3), 

    #     Dense(50, activation='softmax')
    # ])


    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # epoche = 30

    # model.summary()

    # early_stopping = EarlyStopping(
    #         monitor='val_accuracy',
    #         patience=5,
    #         restore_best_weights=True
    #     )
    
    # history = model.fit(train_set, epochs=epoche, validation_data=valid_set, batch_size=batch_size,callbacks = early_stopping)

    # # Accuratezza
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Accuracy over epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # # Loss
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Loss over epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


    # model.save(filepath=r'.\model\lego.keras')
    # print('Modello salvato')
    
    model = load_model(r'./model/lego.keras')

    # etichette previste
    y_hat_pre = model.predict(test_set)
    print("Predictions:\n", y_hat_pre)
    
    y_hat = np.argmax(y_hat_pre, axis=1)
    print("Predictions:\n", y_hat)
    
    print(y_hat.shape)
    
    #etichette reali 
    y_true = train_set.classes
    
    
    loss,accuracy = model.evaluate(test_set)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    
    # Da qui in poi solo grafici 
    cm = confusion_matrix(y_true, y_hat)

    # Visualizzazione della matrice di confusione
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_set.classes.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    

    # print(type(model))
    # print(train_set.image_shape)
    # print(test_set.image_shape)
    # print(valid_set.image_shape)

    # # aggiunge la label di ogni classe in un array 
    # class_names =[]
    # for nome_cartella in os.listdir(r'Data\Train_set'):
    #     class_names.append(nome_cartella)
    
    # images = np.concatenate([batch[0] for batch in test_set], axis=0)
    # labels = np.concatenate([batch[1] for batch in test_set], axis=0)

    # print("Images shape:", images.shape)
    # print("Labels shape:", labels.shape)

    # # y_hat = model.predict(images)
    # # print("Predictions:", y_hat)


    # for images, labels in zip(images,labels):
    #     y_hat = model.predict(images)
    #     print("Predictions:\n", y_hat)
    #     print("True Labels:\n", labels)

    # plt.figure(figsize=(10, 10))

    # for images, labels in zip(images,labels):   
    #     y_hat = model.predict(images)
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].astype("uint8"), cmap='gray')
    #         plt.title(f"Pred: {class_names[np.argmax(y_hat[i])]} \n True: {class_names[np.argmax(labels[i])]}")
    #         plt.axis("off")
            
    # plt.show()
    
