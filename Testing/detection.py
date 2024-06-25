import cv2
import numpy as np
import tensorflow as tf
from keras import models as mdl
import matplotlib.pyplot as plt


image_path = r'img\lego.jpg'


img = cv2.imread(image_path)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


img_resized = cv2.resize(img, (200, 200))


img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)


img_normalized = img_gray / 255.0


img_batch = np.expand_dims(img_normalized, axis=0)
img_batch = np.expand_dims(img_batch, axis=-1)


model = mdl.load_model(r'.\model\lego.keras')


predictions = model.predict(img_batch)


predicted_class = np.argmax(predictions, axis=-1)
print(f'Predicted class: {predicted_class[0]}')


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Predicted Class: {predicted_class[0]}')
plt.show()
