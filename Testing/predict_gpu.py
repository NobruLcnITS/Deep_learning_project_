import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
import seaborn as sns

if __name__ == '__main__':
        
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_set_path = r"./Data/Train_set"
    test_set_path = r"./Data/Test_set"
    valid_set_path = r"./Data/Val_set"
    
    image_size = (400, 400)
    batch_size = 32
    
    datagen_2 = ImageDataGenerator(rescale=1./255)
    
    test_set = datagen_2.flow_from_directory(
        directory=test_set_path,
        class_mode='categorical',
        color_mode="grayscale",
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
        interpolation="bilinear"
    )
    
    model = load_model(r'./model/lego.keras')

    # Previsioni batch-by-batch
    y_hat_pre = model.predict(test_set)
    y_hat = np.argmax(y_hat_pre, axis=1)
    
    # Etichette reali 
    y_true = test_set.classes

    loss, accuracy = model.evaluate(test_set)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    
    # Matrice di confusione
    cm = confusion_matrix(y_true, y_hat)

    class_labels = list(test_set.class_indices.keys())
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    
    # Impostazioni per migliorare la leggibilità delle etichette sull'asse x
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()
