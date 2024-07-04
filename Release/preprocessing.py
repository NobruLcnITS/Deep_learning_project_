import pandas as pd
import numpy as np
import glob
import os
import shutil

# Suddivisione del dataset in sottocartelle per le classi
df = pd.read_csv(r'./validation.csv', header=0)


percorso_dataset = r"./Data\dataset"
percorso_dataset_corretto = r"./Data\Dataset_corretto"
percorso_dataset_train = r"./Data\Train_set"
percorso_dataset_test = r"./Data\Test_set"
percorso_dataset_val = r"./Data\Val_set"

df_selected = df.groupby('Id').first()
print(df_selected)

labels = df_selected['Description']
ids = df_selected.index.values

# print(len(labels),len(ids))

# print(labels)

# creazione dataset con le sottocartelle come classi per ogni tipo di brick

for label,id in zip(labels,ids):
    os.makedirs(f"{percorso_dataset_corretto}/{id} {label}", exist_ok=True)

for nome_cartella in os.listdir(percorso_dataset_corretto):
    for nome_file in os.listdir(f"{percorso_dataset}"):
        
        if nome_cartella in nome_file:
            shutil.copy(f"{percorso_dataset}/{nome_file}", f"{percorso_dataset_corretto}/{nome_cartella}")
            
    
# conta qunate sottocartelle ci sono nel datset corretto
cout= len(os.listdir(percorso_dataset_corretto))

print(cout)


# creazione delle cartelle train e test
   
train_size = 0.8
test_size = 0.1
validation_size = 0.1


for label,id in zip(labels,ids):
    os.makedirs(f"{percorso_dataset_train}/{id} {label}", exist_ok=True)
    os.makedirs(f"{percorso_dataset_test}/{id} {label}", exist_ok=True)
    os.makedirs(f"{percorso_dataset_val}/{id} {label}", exist_ok=True)
    
 
for nome_cartella in os.listdir(percorso_dataset_train):
    immagini = os.listdir(f"{percorso_dataset_corretto}/{nome_cartella}")
    # splitta il train
    for nome_file in immagini[:int(len(immagini)*train_size)]:
            shutil.copy(f"{percorso_dataset_corretto}/{nome_cartella}/{nome_file}", f"{percorso_dataset_train}/{nome_cartella}")
            
    # splitta il test              
    for nome_file in immagini[int(len(immagini)*train_size):int(len(immagini)*(1-test_size))]:
            shutil.copy(f"{percorso_dataset_corretto}/{nome_cartella}/{nome_file}", f"{percorso_dataset_test}/{nome_cartella}")

    # splitta il validation  
    for nome_file in immagini[int(len(immagini)*(1-validation_size)):]:
            shutil.copy(f"{percorso_dataset_corretto}/{nome_cartella}/{nome_file}", f"{percorso_dataset_val}/{nome_cartella}")
    

# conta quante immagini ci sono per ogni suddivisione
count = 0

print('TRAIN')
for nome_cartella in os.listdir(percorso_dataset_train):
    for nome_file in os.listdir(f"{percorso_dataset_train}/{nome_cartella}"):
        count +=1
    print(f'cartella {nome_cartella} : ',count)
    count=0

print('TEST')
for nome_cartella in os.listdir(percorso_dataset_test):
    for nome_file in os.listdir(f"{percorso_dataset_test}/{nome_cartella}"):
        count +=1
    print(f'cartella {nome_cartella} : ',count)
    count=0
    
print('VALIDATION')
for nome_cartella in os.listdir(percorso_dataset_val):
    for nome_file in os.listdir(f"{percorso_dataset_val}/{nome_cartella}"):
        count +=1
    print(f'cartella {nome_cartella} : ',count)
    count=0