#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split



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

#  Define modification percentages
modification_percentages = [0, 10, 20, 40]

#  Store results in a table
results = []

#  Store confusion matrices for visualization
conf_matrices = []


#  Function to modify data (corrupt temperature values)
def modify_data(df, percentage):
    df_corrupted = df.copy()
    num_samples = int(len(df) * (percentage / 100))
    
    # Select random indices to corrupt
    indices = np.random.choice(df.index, size=num_samples, replace=False)
    
    # Corrupt temperature data by adding random noise
    noise = np.random.uniform(-5, 5, size=num_samples)  # Adjust noise range as needed
    df_corrupted.loc[indices, "temp"] += noise

    return df_corrupted

#  Function to modify labels (flip anomaly labels)
def flip_labels(df, percentage):
    df_flipped = df.copy()

      
    num_samples = int(len(df_flipped) * (percentage / 100))

    # Select random indices to flip labels
    indices = np.random.choice(df_flipped.index, size=num_samples, replace=False)

    # Flip 0 ↔ 1
    df_flipped.loc[indices, "anomaly"] = 1 - df_flipped.loc[indices, "anomaly"]

    return df_flipped


#  Function to train and evaluate LSTM
def train_and_evaluate(df, modification_type, mod_percentage):
    # Select features
    selected_columns = ["day", "month", "time", "temp", "location_encoded"]
    data_selected = df[selected_columns].dropna()

    # Define anomalies based on extreme temperature (fixed threshold)
    data_selected['anomaly'] = ((df['temp'] < df['temp'].quantile(0.05)) | 
                                (df['temp'] > df['temp'].quantile(0.97))).astype(int)

    # Normalize temperature values
    scaler = MinMaxScaler()
    data_selected["temp_normalized"] = scaler.fit_transform(data_selected["temp"].values.reshape(-1, 1))

    # Prepare LSTM sequences
    sequence_length = 4
    X, y = [], []

    for i in range(len(data_selected) - sequence_length):
        X.append(data_selected["temp_normalized"].values[i:i + sequence_length])
        y.append(data_selected["anomaly"].values[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define LSTM Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate Model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)
    predictions_binary = (predictions > 0.5).astype(int)

    # Compute Metrics
    precision = precision_score(y_test, predictions_binary)
    recall = recall_score(y_test, predictions_binary)
    f1 = f1_score(y_test, predictions_binary)
    roc_auc = roc_auc_score(y_test, predictions_binary)
    conf_matrix = confusion_matrix(y_test, predictions_binary)

    print(f"\n {modification_type} {mod_percentage}% - Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}\n")

    return [mod_percentage, accuracy, precision, recall, f1, roc_auc], (mod_percentage, conf_matrix)

#  Store results
results_data_mod = []
results_label_mod = []

# Store confusion matrices
conf_matrices_data_mod = []
conf_matrices_label_mod = []

#  Apply and evaluate for different modification levels
for mod_percentage in modification_percentages:
    print(f"\n Evaluating Data Modification {mod_percentage}%...\n")
    df_data_mod = modify_data(df, mod_percentage)
    res_data, conf_data = train_and_evaluate(df_data_mod, "Data Modification", mod_percentage)
    results_data_mod.append(res_data)
    conf_matrices_data_mod.append(conf_data)

    print(f"\n Evaluating Label Modification {mod_percentage}%...\n")
    df_label_mod = flip_labels(df, mod_percentage)
    res_label, conf_label = train_and_evaluate(df_label_mod, "Label Modification", mod_percentage)
    results_label_mod.append(res_label)
    conf_matrices_label_mod.append(conf_label)

#  Convert results to DataFrame
df_results_data = pd.DataFrame(results_data_mod, columns=["Modification %", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"])
df_results_label = pd.DataFrame(results_label_mod, columns=["Modification %", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"])

print("\nEvaluation Results (Data Modification):\n", df_results_data)
print("\nEvaluation Results (Label Modification):\n", df_results_label)

#  Plot confusion matrices in a single row for Data Modification
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Confusion Matrices for Data Modification Levels")

for i, (mod_percentage, matrix) in enumerate(conf_matrices_data_mod):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=['0', '1'], yticklabels=['0', '1'])
    axes[i].set_title(f"{mod_percentage}% Modification")
    axes[i].set_xlabel("Predicted Labels")
    axes[i].set_ylabel("True Labels")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot confusion matrices in a single row for Label Modification
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Confusion Matrices for Label Modification Levels")

for i, (mod_percentage, matrix) in enumerate(conf_matrices_label_mod):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=['0', '1'], yticklabels=['0', '1'])
    axes[i].set_title(f"{mod_percentage}% Modification")
    axes[i].set_xlabel("Predicted Labels")
    axes[i].set_ylabel("True Labels")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()







