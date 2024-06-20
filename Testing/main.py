import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn  
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer
from keras import layers, metrics, optimizers, models, losses, utils
from keras.preprocessing import image_dataset_from_directory
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np  
from PIL import Image
import glob


print('Processing....')

dir = r"./Data\LEGO brick images v1"
df_train,df_test= image_dataset_from_directory(
    dir,
    color_mode="grayscale",
    image_size=(200, 200),
    batch_size = 128,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    validation_split=0.2,
    subset='both'
)


print(df_train.class_names)

print('Fine')







