#!/usr/bin/env python
# coding: utf-8

# In[1]:


#3 EDA sur les relevés de température Dispositifs iot
# OU accès aux données des températures lues par les dispositifs IoT
#1 Importation de la librairie
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
#!pip install tensorflow
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[3]:


#Lecture les fichiers de données et fournir des informations de base
data = pd.read_csv("C:/Users/fvuse/OneDrive/Documents/IOT_temp.csv")
# Affichage de 5 lignes aléatoires
data.sample(5)


# In[4]:


# Affichage de 5 premières lignes
data.head()


# In[5]:


# Les 5 dernières lignes
data.tail()


# In[6]:


#La forme de nos données est la suivante :
print("La forme de nos données est: ",data.shape)


# In[7]:


#Les Valeurs uniques dans chaque colonne sont les suivantes:
print("Valeurs uniques dans chaque colonne \n"+'-'*25)
for i in data.columns:
    print("\t"+i+" = ",len(set(data[i])))


# In[8]:


# Compter le nombre de valeurs "X" dans la colonne
nombre_X = data['out/in'].value_counts()['Out']

print(nombre_X)


# In[9]:


#Les informations sur les caractéristiques de données
data.info()


# In[10]:


# Description des données
data.describe()


# In[11]:


# Suppression des données dont le label est 'id' et 'room_id/id'
df = data.drop(['id','room_id/id'],axis=1)
df.head()


# In[12]:


# 2ème partie : Analyse des données
#Vérification des valeurs manquantes
#a) Données qui sont nulles
data.isnull().sum()


# In[13]:


#b)Sépareration de la date et l'heure
date=[]
time=[]
for i in df['noted_date']:
    date.append(i.split(' ')[0])
    time.append(i.split(' ')[1])
df['date']=date
df['time']=time


# In[14]:


# Suppression du label 'note_date'
df.drop('noted_date',axis=1,inplace=True)
df.head()


# In[15]:


df[['outside','inside']]=pd.get_dummies(df['out/in'])
df.rename(columns = {'out/in':'location'}, inplace = True)


# In[16]:


print('Total Inside Observations  :',len([i for i in df['inside'] if  i == 1]))
print('Total Outside Observations :',len([i for i in df['inside'] if  i == 0]))


# In[ ]:





# In[17]:


print(df.columns)


# In[ ]:





# In[18]:


df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", dayfirst=True)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df.drop('date',axis=1,inplace=True)

print('Operations already performed')
df.head()


# In[19]:


print("Days of observation   : ",sorted(df['day'].unique()))
print("Months of observation : ",sorted(df['month'].unique()))
print("Year of observation   : ",sorted(df['year'].unique()))


# In[20]:


# Les valeurs des données des températures
print("Temperature -> \n"+"-"*30)
print("\tTotal Count    = ",df['temp'].shape[0])
print("\tMinimum Value  = ",df['temp'].min())
print("\tMaximum Value  = ",df['temp'].max())
print("\tMean Value     = ",df['temp'].mean())
print("\tStd dev Value  = ",df['temp'].std())
print("\tVariance Value = ",df['temp'].var())


# In[21]:


#Réassemblage de la base de données et affichage de la nouvelle base de données détaillée
df = df[['day','month','year','time','temp','location']]
df.head()


# In[22]:


nombre_noeuds = int(input("Entrez le nombre de nœuds à sélectionner : "))


# In[23]:


# Affichage du nombre de noeuds selectionnées
print("Les noeuds pris aléatoirement sont au nombre de : ",nombre_noeuds )


# In[24]:


import random


# In[25]:


indices_noeuds = random.sample(range(len(df)), nombre_noeuds)


# In[26]:


noeuds_selectionnes = df.iloc[indices_noeuds]


# In[27]:


# Affichage des nœuds sélectionnés
print("Voici les noeuds selectionnés")
print("Noeuds", noeuds_selectionnes)


# In[28]:


print(data.columns)


# In[29]:


print(df.columns)


# In[30]:


# Encodage de la localisation
df['location_encoded'] = df['location'].astype('category').cat.codes


# In[31]:


df["time"] = pd.to_datetime(df["time"], format="%H:%M").dt.hour * 60 + pd.to_datetime(df["time"], format="%H:%M").dt.minute


# In[32]:


df.head()


# In[ ]:





# In[33]:


import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 Ensure 'anomaly' column exists
if "anomaly" not in df.columns:
    df["anomaly"] = ((df["temp"] < df["temp"].quantile(0.05)) | (df["temp"] > df["temp"].quantile(0.97))).astype(int)

# 🔹 Function to Modify Data
def modify_data(df, percentage):
    df_modified = df.copy()
    num_samples = int(len(df_modified) * (percentage / 100))
    indices = np.random.choice(df_modified.index, size=num_samples, replace=False)

    # Modify temperature values randomly
    df_modified.loc[indices, "temp"] += np.random.uniform(-5, 5, size=num_samples)
    
    return df_modified

# 🔹 Function to Flip Labels (Label Modification)
def flip_labels(df, percentage):
    df_flipped = df.copy()
    num_samples = int(len(df_flipped) * (percentage / 100))
    indices = np.random.choice(df_flipped.index, size=num_samples, replace=False)

    # Flip 0 ↔ 1
    df_flipped.loc[indices, "anomaly"] = 1 - df_flipped.loc[indices, "anomaly"]

    return df_flipped

# 🔹 Function to Train and Evaluate One-Class SVM
def train_and_evaluate(df, modification_type, percentage):
    selected_columns = ["day", "month", "temp", "location_encoded", "anomaly"]
    data_selected = df[selected_columns].dropna()

    # 🔹 Normalization
    scaler = MinMaxScaler()
    data_selected["temp_normalized"] = scaler.fit_transform(data_selected["temp"].values.reshape(-1, 1))

    # 🔹 Extract Features and Labels
    X = data_selected[["day", "month", "temp_normalized", "location_encoded"]].values
    y = data_selected["anomaly"].values

    # 🔹 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 🔹 Train One-Class SVM Model
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)  # nu = expected anomaly rate (5%)
    model.fit(X_train)

    # 🔹 Predict Anomalies
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)  # Convert One-Class SVM output (-1 = anomaly, 1 = normal) to (1 = anomaly, 0 = normal)

    # 🔹 Compute Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc_roc   = roc_auc_score(y_test, y_pred)

    # 🔹 Compute Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print Results
    print(f"\n📌 Results for {modification_type} - {percentage}% Modification:")
    print(f"✅ Accuracy  : {accuracy:.4f}")
    print(f"✅ Precision : {precision:.4f}")
    print(f"✅ Recall    : {recall:.4f}")
    print(f"✅ F1-score  : {f1:.4f}")
    print(f"✅ AUC-ROC   : {auc_roc:.4f}")

    return [modification_type, percentage, accuracy, precision, recall, f1, auc_roc], conf_matrix

# 🔹 Define Modification Percentages
modification_percentages = [0, 10, 20, 40]

# 🔹 Store Results
results_data_mod = []
results_label_mod = []
conf_matrices_data_mod = []
conf_matrices_label_mod = []

# 🔹 Evaluate Data Modification
for mod_percentage in modification_percentages:
    df_data_mod = modify_data(df, mod_percentage)
    res_data, conf_data = train_and_evaluate(df_data_mod, "Data Modification", mod_percentage)
    results_data_mod.append(res_data)
    conf_matrices_data_mod.append((mod_percentage, conf_data))

# 🔹 Evaluate Label Modification
for mod_percentage in modification_percentages:
    df_label_mod = flip_labels(df, mod_percentage)
    res_label, conf_label = train_and_evaluate(df_label_mod, "Label Modification", mod_percentage)
    results_label_mod.append(res_label)
    conf_matrices_label_mod.append((mod_percentage, conf_label))

# 🔹 Convert Results to DataFrame
columns = ["Modification Type", "Percentage", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
df_results = pd.DataFrame(results_data_mod + results_label_mod, columns=columns)

# 🔹 Display Results
print("\n📊 Final Results Table:")
print(df_results)

# 🔹 Plot Confusion Matrices
fig, axes = plt.subplots(2, len(modification_percentages), figsize=(12, 8))
fig.suptitle("Confusion Matrices for Different Modification Levels")

for i, (mod_percentage, matrix) in enumerate(conf_matrices_data_mod):
    ax = axes[0, i]
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['0', '1'], yticklabels=['0', '1'])
    ax.set_title(f"Data Mod {mod_percentage}%")

for i, (mod_percentage, matrix) in enumerate(conf_matrices_label_mod):
    ax = axes[1, i]
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['0', '1'], yticklabels=['0', '1'])
    ax.set_title(f"Label Mod {mod_percentage}%")

plt.tight_layout()
plt.show()



# In[ ]:





# In[ ]:




