import glob
from os.path import basename
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def conta_immagini(image_path: str) -> int:
    return len(glob.glob(f'{image_path}/*.png'))

# Carica le immagini e le etichette
def carica_immagini(image_folder: str):
    image_paths = glob.glob(f'{image_folder}/*.png')
    images = []
    labels = []
    for path in image_paths:
        img = Image.open(path)
        img_array = np.array(img)
        images.append(img_array)
        
        # Estrai la label dal nome del file, se necessario
        label = basename(path).split('_')[0]  # Questo è solo un esempio, modifica secondo il tuo caso
        labels.append(label)
    
    return np.array(images), np.array(labels)

train_folder = r'./Data/dataset'

# Conta il numero di immagini nella cartella
n_train = conta_immagini(train_folder)
print(f'Numero di immagini nel dataset: {n_train}')

print('Caricamento immagini...')
x_data, y_data = carica_immagini(train_folder)

# Dividi i dati in set di addestramento e di test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Esempio per visualizzare una delle immagini di addestramento
Image.fromarray(x_train[0]).show()
print('Fine caricamento')
