#!/usr/bin/env python
# coding: utf-8

#1 Library import
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#Read data files and provide basic information
data = pd.read_csv("C:/Users/fvuse/OneDrive/Documents/IOT_temp.csv")
#  Display of 5 random lines
data.sample(5)

# Display of first 5 lines
data.head()

# The last 5 lines
data.tail()

#The form of our data is as follows:
print("Shape of our data is : ",data.shape)

#The unique Values in each column are as follows:
print("Unique values in every column \n"+'-'*25)
for i in data.columns:
    print("\t"+i+" = ",len(set(data[i])))

# Count the number of "X" values in the column
nombre_X = data['out/in'].value_counts()['Out']

print(nombre_X)

#Information on data characteristics
data.info()

# Data description
data.describe()

# Deletion of data with 'id' and 'room_id/id' labels
df = data.drop(['id','room_id/id'],axis=1)
df.head()

#Data analysis
#Checking for missing values
#a) Null data
data.isnull().sum()


df[['outside','inside']]=pd.get_dummies(df['out/in'])

print('Total Inside Observations  :',len([i for i in df['inside'] if  i == 1]))
print('Total Outside Observations :',len([i for i in df['inside'] if  i == 0]))

# Temperature data values
print("Temperature -> \n"+"-"*30)
print("\tTotal Count    = ",df['temp'].shape[0])
print("\tMinimum Value  = ",df['temp'].min())
print("\tMaximum Value  = ",df['temp'].max())
print("\tMean Value     = ",df['temp'].mean())
print("\tStd dev Value  = ",df['temp'].std())
print("\tVariance Value = ",df['temp'].var())


#Reassembling the database and displaying the new detailed database
df = df[['noted_date','temp','out/in','outside','inside']]
df.head()


nomber_nodes = int(input("Enter the number of nodes to select : "))


indices_nodes = random.sample(range(len(df)), nomber_nodes)


select_nodes = df.iloc[indices_nodes]


# Display selected nodes
print("Here are the selected nodes")
print("Nodes", select_nodes)


print(data.columns)

print(df.columns)


# Location encoding
df['location_encoded'] = df['location'].astype('category').cat.codes



df["time"] = pd.to_datetime(df["time"], format="%H:%M").dt.hour * 60 + pd.to_datetime(df["time"], format="%H:%M").dt.minute


df.head()


#  Creating the anomaly column
df["anomaly"] = ((df["temp"] < df["temp"].quantile(0.05)) | (df["temp"] > df["temp"].quantile(0.97))).astype(int)


#  Function to Modify Data
def modify_data(df, percentage):
    df_modified = df.copy()
    num_samples = int(len(df_modified) * (percentage / 100))
    indices = np.random.choice(df_modified.index, size=num_samples, replace=False)

    # Modify temperature values randomly
    df_modified.loc[indices, "temp"] += np.random.uniform(-5, 5, size=num_samples)
    
    return df_modified

#  Function to Flip Labels (Label Modification)
def flip_labels(df, percentage):
    df_flipped = df.copy()
    num_samples = int(len(df_flipped) * (percentage / 100))
    indices = np.random.choice(df_flipped.index, size=num_samples, replace=False)

    # Flip 0 ↔ 1
    df_flipped.loc[indices, "anomaly"] = 1 - df_flipped.loc[indices, "anomaly"]

    return df_flipped

#  Function to Train and Evaluate Isolation Forest
def train_and_evaluate(df, modification_type, percentage):
    selected_columns = ["day", "month", "temp", "location_encoded", "anomaly"]
    data_selected = df[selected_columns].dropna()

    #  Normalization
    scaler = MinMaxScaler()
    data_selected["temp_normalized"] = scaler.fit_transform(data_selected["temp"].values.reshape(-1, 1))

    #  Extract Features and Labels
    X = data_selected[["day", "month", "temp_normalized", "location_encoded"]].values
    y = data_selected["anomaly"].values

    #  Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #  Train Isolation Forest Model
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X_train)

    #  Predict Anomalies
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)  # Convert Isolation Forest output (-1 = anomaly, 1 = normal) to (1 = anomaly, 0 = normal)

    #  Compute Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc_roc   = roc_auc_score(y_test, y_pred)

    #  Compute Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print Results
    print(f"\n Results for {modification_type} - {percentage}% Modification:")
    print(f" Accuracy  : {accuracy:.4f}")
    print(f" Precision : {precision:.4f}")
    print(f" Recall    : {recall:.4f}")
    print(f" F1-score  : {f1:.4f}")
    print(f" AUC-ROC   : {auc_roc:.4f}")

    return [modification_type, percentage, accuracy, precision, recall, f1, auc_roc], conf_matrix

#  Define Modification Percentages
modification_percentages = [0, 10, 20, 40]

#  Store Results
results_data_mod = []
results_label_mod = []
conf_matrices_data_mod = []
conf_matrices_label_mod = []

#  Evaluate Data Modification
for mod_percentage in modification_percentages:
    df_data_mod = modify_data(df, mod_percentage)
    res_data, conf_data = train_and_evaluate(df_data_mod, "Data Modification", mod_percentage)
    results_data_mod.append(res_data)
    conf_matrices_data_mod.append((mod_percentage, conf_data))

#  Evaluate Label Modification
for mod_percentage in modification_percentages:
    df_label_mod = flip_labels(df, mod_percentage)
    res_label, conf_label = train_and_evaluate(df_label_mod, "Label Modification", mod_percentage)
    results_label_mod.append(res_label)
    conf_matrices_label_mod.append((mod_percentage, conf_label))

#  Convert Results to DataFrame
columns = ["Modification Type", "Percentage", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
df_results = pd.DataFrame(results_data_mod + results_label_mod, columns=columns)

#  Display Results
print("\n Final Results Table:")
print(df_results)

#  Plot Confusion Matrices
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




