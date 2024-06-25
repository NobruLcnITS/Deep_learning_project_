import pandas as pd
import numpy as np
import glob
import os
import shutil

df = pd.read_csv(r'./Data\validation.csv', header=0)


percorso_dataset_corretto = "Data\Dataset_corretto"
percorso_dataset = "Data\dataset"

df_selected = df.groupby('Id').first()
print(df_selected)

labels = df_selected['Description']
ids = df_selected.index.values

print(len(labels),len(ids))

print(labels)

for label,id in zip(labels,ids):
    os.makedirs(f"{percorso_dataset_corretto}/{id} {label}", exist_ok=True)

for nome_cartella in os.listdir(percorso_dataset_corretto):
    for nome_file in os.listdir(f"{percorso_dataset}"):
        
        if nome_cartella in nome_file:
            shutil.copy(f"{percorso_dataset}/{nome_file}", f"{percorso_dataset_corretto}/{nome_cartella}")
            
    
cout= len(os.listdir(percorso_dataset_corretto))

print(cout)