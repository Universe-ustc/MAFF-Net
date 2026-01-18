import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer

lstm_model_path = r"E:\新大空间视频\lstm_model_combined.pth"
data_path = r"E:\新大空间视频\训练传感器数据\2024-01-26_11-21-24_3_0.75_3.xlsx"
image_folder_path = "2024-01-26_11-23-17_11-31-09_3_0.75"
resnet_model_path = "resnet18_model.pth"

def read_sensor_data(sheet_number):
    df = pd.read_excel(data_path, sheet_name=sheet_number)  
    co2_data = df.iloc[:, 2].values.astype(float)
    tvoc_data = df.iloc[:, 4].values.astype(float)
    temperature_data = df.iloc[:, 6].values.astype(float)
    humidity_data = df.iloc[:, 8].values.astype(float)
    smoke_data = df.iloc[:, 10].values.astype(float)
    co_data = df.iloc[:, 12].values.astype(float)
    fire_label = df.iloc[:, -1].str.strip().replace({'notfire': 0, 'fire': 1}).values.astype(int)

    imputer = SimpleImputer(strategy='mean')
    co2_data = imputer.fit_transform(co2_data.reshape(-1, 1))
    tvoc_data = imputer.fit_transform(tvoc_data.reshape(-1, 1))
    temperature_data = imputer.fit_transform(temperature_data.reshape(-1, 1))
    humidity_data = imputer.fit_transform(humidity_data.reshape(-1, 1))
    smoke_data = imputer.fit_transform(smoke_data.reshape(-1, 1))
    co_data = imputer.fit_transform(co_data.reshape(-1, 1))

    scaler = MinMaxScaler()
    co2 = scaler.fit_transform(co2_data.reshape(-1, 1))
    tvoc = scaler.fit_transform(tvoc_data.reshape(-1, 1))
    temperature = scaler.fit_transform(temperature_data.reshape(-1, 1))
    humidity = scaler.fit_transform(humidity_data.reshape(-1, 1))
    smoke = scaler.fit_transform(smoke_data.reshape(-1, 1))
    co = scaler.fit_transform(co_data.reshape(-1, 1))

    sensor_data = np.column_stack((co2, tvoc, temperature, humidity, smoke, co))
    return sensor_data, fire_label

def read_image(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    image = Image.open(file_path).convert('RGB')  
    return image

class MyModel(nn.Module):
    def __init__(self, input_dimension, hidden_units, num_layers, num_classes):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_dimension, hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, num_classes) 
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  
        return x

input_dimension = 6  
hidden_units = 32
num_layers = 2  
num_classes = 2  
lstm_model = MyModel(input_dimension, hidden_units, num_layers, num_classes)
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device('cpu')))
lstm_model.eval()

resnet_model = models.resnet18(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 2)
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=torch.device('cpu')))
resnet_model.eval()

preprocess = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_features = []
for i in range(473):
    folder = "notfire" if i < 38 else "fire"
    filename = f"{image_folder_path}\\{folder}\\frame_{i}.jpg"
    image = read_image(filename)
    if image is not None:
        image = preprocess(image)
        image_features.append(image)

image_features = torch.stack(image_features)
image_features = resnet_model(image_features).detach().numpy()

clf = RandomForestClassifier(n_estimators=100)
labels = ["notfire"] * 38 + ["fire"] * 435
labels_int = [0 if label == 'notfire' else 1 for label in labels]
clf.fit(image_features, labels_int)
importances = clf.feature_importances_
threshold = np.mean(importances)
important_indices = np.where(importances > threshold)[0]
important_features = image_features[:, important_indices]

distances = range(1, 7)
accuracies = []
precisions = []
recalls = []
f1_scores = []

for sheet_number in distances:
    sensor_data, fire_label = read_sensor_data(sheet_number)

    X_tensor = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)  

    with torch.no_grad():
        lstm_outputs, _ = lstm_model.lstm(X_tensor) 
        sensor_features_full = lstm_outputs.squeeze(0).numpy()

    X_tensor.requires_grad = True
    outputs, _ = lstm_model.lstm(X_tensor)  
    loss = outputs.mean()
    loss.backward()
    gradients = X_tensor.grad.data.numpy().squeeze(0)
    weights = np.mean(np.abs(gradients), axis=1)
    threshold = np.percentile(weights[:-120], 95) 

    key_time_steps = np.where(weights > threshold)[0]
    key_time_steps = key_time_steps[key_time_steps < sensor_features_full.shape[0]]

    weight_multiplier = 2
    sensor_features_weighted = sensor_features_full.copy()
    sensor_features_weighted[key_time_steps] *= weight_multiplier

    n_components = sensor_features_weighted.shape[1]
    pca = KernelPCA(n_components=n_components)
    image_features_reduced = pca.fit_transform(important_features)

    num_features = min(sensor_features_weighted.shape[0], image_features_reduced.shape[0])
    sensor_features_weighted = sensor_features_weighted[:num_features]
    image_features_reduced = image_features_reduced[:num_features]

    fused_features = np.concatenate((sensor_features_weighted, image_features_reduced), axis=1)

    pca_final = KernelPCA(n_components=2)
    merged_features = pca_final.fit_transform(fused_features)

    svm_model = SVC(probability=True)

    svm_model.fit(merged_features, labels_int[:num_features])

    predictions = svm_model.predict(merged_features)

    accuracy = accuracy_score(labels_int[:num_features], predictions)
    precision = precision_score(labels_int[:num_features], predictions)
    recall = recall_score(labels_int[:num_features], predictions)
    f1 = f1_score(labels_int[:num_features], predictions)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

plt.figure(figsize=(12, 8))
plt.plot(distances, accuracies, marker='o', linestyle='-', color='#4169E1', label='Accuracy', linewidth=2.5, markersize=10)
plt.plot(distances, precisions, marker='s', linestyle='--', color='orange', label='Precision', linewidth=2.5, markersize=10)
plt.plot(distances, recalls, marker='^', linestyle='-.', color='green', label='Recall', linewidth=2.5, markersize=10)
plt.plot(distances, f1_scores, marker='d', linestyle=':', color='red', label='F1-score', linewidth=2.5, markersize=10)
plt.xlabel('Distance (m)', fontsize=22)
plt.ylabel('Score', fontsize=22)
plt.title('Performance Metrics vs Distance', fontsize=24)
plt.legend(loc='lower left', fontsize=22, frameon=True, shadow=True, borderpad=1)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)  
plt.tight_layout()
plt.savefig('performance_vs_distance_3.png', dpi=300)
plt.show()




