import cv2
import numpy as np
import tensorflow as tf
from keras import models as mdl
import matplotlib.pyplot as plt

# Percorso dell'immagine
image_path = r'img\lego.jpg'

# Caricare l'immagine usando OpenCV
img = cv2.imread(image_path)

# Mostrare l'immagine originale
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Ridimensionare l'immagine alla dimensione richiesta dal modello
img_resized = cv2.resize(img, (200, 200))

# Convertire l'immagine in scala di grigi se necessario
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Normalizzare l'immagine (opzionale, dipende dal modello)
img_normalized = img_gray / 255.0

# Aggiungere un asse per il batch (se il modello si aspetta un batch di immagini)
img_batch = np.expand_dims(img_normalized, axis=0)
img_batch = np.expand_dims(img_batch, axis=-1)  # Aggiungere un asse per i canali se richiesto

# Caricare il modello addestrato
model = mdl.load_model('path_to_your_model.h5')

# Fare una previsione
predictions = model.predict(img_batch)

# Interpretare il risultato (dipende dal tipo di modello e dal problema)
predicted_class = np.argmax(predictions, axis=-1)
print(f'Predicted class: {predicted_class[0]}')

# Visualizzare il risultato
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Predicted Class: {predicted_class[0]}')
plt.show()
