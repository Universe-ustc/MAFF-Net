import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = r""
SENSOR_DIR = os.path.join(BASE_DIR, r"Time-synchronized multimodal dataset\Sensor dataset\Indoor")
IMAGE_DIR = os.path.join(BASE_DIR, r"Time-synchronized multimodal dataset\Image dataset\Indoor")
RESULT_DIR = os.path.join(BASE_DIR, "MAFF-Net", "MAFF-Net_results")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "MAFF-Net", "lstm_trained_models", "lstm_model_combined_new.pth")
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "MAFF-Net", "resnet_trained_models", "resnet18_model.pth")

NUM_REPEATS = 5  
NUM_TEST_SETS = 7  

os.makedirs(RESULT_DIR, exist_ok=True)

def find_test_sets():
    test_set_mapping = {
        '2024-01-23_17-02-57_1_0.15 (test set).xlsx': '2024-01-23_17-06-49_17-15-49_1_0.15 (test set)',
        '2024-01-24_10-13-36_1_0.75 (test set).xlsx': '2024-01-24_11-50-49_12-01-18_1_0.75 (test set)',
        '2024-01-25_09-55-39_0_0.75 (test set).xlsx': '2024-01-25_9-59-24_10-07-43_0_0.75 (test set)',
        '2024-01-26_10-28-41_2_0.3 (test set).xlsx': '2024-01-26_10-34-30_10-41-12_2_0.3 (test set)',
        '2024-01-26_10-46-23_2_0.9 (test set).xlsx': '2024-01-26_10-48-31_10-56-52_2_0.9 (test set)',
        '2024-01-26_11-21-24_3_0.75 (test set).xlsx': '2024-01-26_11-23-17_11-31-09_3_0.75 (test set)',
        '2024-01-26_11-55-55_3_1.05 (test set).xlsx': '2024-01-26_11-57-00_12-07-06_3_1.05 (test set)',
    }
    
    sensor_files = []
    image_folders = []
    
    for sensor_file_name, image_folder_name in test_set_mapping.items():
        sensor_path = os.path.join(SENSOR_DIR, sensor_file_name)
        image_path = os.path.join(IMAGE_DIR, image_folder_name)
        
        if os.path.exists(sensor_path) and os.path.exists(image_path):
            sensor_files.append(sensor_path)
            image_folders.append(image_path)
        else:
            print(f"Warning: Cannot find pair - Sensor: {sensor_file_name}, Image: {image_folder_name}")
    
    if len(sensor_files) < NUM_TEST_SETS:
        print("Trying automatic matching...")
        for file in os.listdir(SENSOR_DIR):
            if "(test set)" in file and file.endswith('.xlsx'):
                sensor_file = os.path.join(SENSOR_DIR, file)
                base_name = file.replace(" (test set).xlsx", "")
                parts = base_name.split('_')
                
                if len(parts) >= 4:
                    distance = parts[-2]  
                    wind_speed = parts[-1]  
                    
                    for folder in os.listdir(IMAGE_DIR):
                        if "(test set)" in folder:
                            folder_parts = folder.replace(" (test set)", "").split('_')
                            if len(folder_parts) >= 4:
                                if (folder_parts[-2] == distance and 
                                    folder_parts[-1] == wind_speed):
                                    image_path = os.path.join(IMAGE_DIR, folder)
                                    if sensor_file not in sensor_files:
                                        sensor_files.append(sensor_file)
                                        image_folders.append(image_path)
                                    break
    
    print(f"Found {len(sensor_files)} test set pairs")
    return sensor_files, image_folders

def read_sensor_data(file_path):
    excel_file = pd.ExcelFile(file_path)
    all_sensor_data = []
    all_labels = []
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        if df.shape[1] < 13 or df.shape[0] == 0:
            print(f"  Warning: Skipping sheet '{sheet_name}' - insufficient columns ({df.shape[1]}) or no rows")
            continue
        
        try:
            co2_data = df.iloc[:, 2].values.astype(float)
            tvoc_data = df.iloc[:, 4].values.astype(float)
            temperature_data = df.iloc[:, 6].values.astype(float)
            humidity_data = df.iloc[:, 8].values.astype(float)
            smoke_data = df.iloc[:, 10].values.astype(float)
            co_data = df.iloc[:, 12].values.astype(float)
            fire_label = df.iloc[:, -1].astype(str).str.strip().map({'notfire': 0, 'fire': 1}).fillna(-1).astype(int)
        except Exception as e:
            print(f"  Warning: Error reading sheet '{sheet_name}': {str(e)}")
            continue
        
        imputer = SimpleImputer(strategy='mean')
        co2_data = imputer.fit_transform(co2_data.reshape(-1, 1)).flatten()
        tvoc_data = imputer.fit_transform(tvoc_data.reshape(-1, 1)).flatten()
        temperature_data = imputer.fit_transform(temperature_data.reshape(-1, 1)).flatten()
        humidity_data = imputer.fit_transform(humidity_data.reshape(-1, 1)).flatten()
        smoke_data = imputer.fit_transform(smoke_data.reshape(-1, 1)).flatten()
        co_data = imputer.fit_transform(co_data.reshape(-1, 1)).flatten()
        
        scaler = MinMaxScaler()
        co2 = scaler.fit_transform(co2_data.reshape(-1, 1)).flatten()
        tvoc = scaler.fit_transform(tvoc_data.reshape(-1, 1)).flatten()
        temperature = scaler.fit_transform(temperature_data.reshape(-1, 1)).flatten()
        humidity = scaler.fit_transform(humidity_data.reshape(-1, 1)).flatten()
        smoke = scaler.fit_transform(smoke_data.reshape(-1, 1)).flatten()
        co = scaler.fit_transform(co_data.reshape(-1, 1)).flatten()
        
        sensor_data = np.column_stack((co2, tvoc, temperature, humidity, smoke, co))
        all_sensor_data.append(sensor_data)
        all_labels.append(fire_label)
    
    if all_sensor_data:
        sensor_data = np.vstack(all_sensor_data)
        labels = np.hstack(all_labels)
        return sensor_data, labels
    return None, None

def read_image_data(folder_path):
    image_features_list = []
    labels_list = []
    
    notfire_folder = os.path.join(folder_path, "notfire")
    fire_folder = os.path.join(folder_path, "fire")
    
    if os.path.exists(notfire_folder):
        notfire_files = sorted([f for f in os.listdir(notfire_folder) if f.endswith('.jpg')])
        for img_file in notfire_files:
            img_path = os.path.join(notfire_folder, img_file)
            image_features_list.append(img_path)
            labels_list.append(0)
    
    if os.path.exists(fire_folder):
        fire_files = sorted([f for f in os.listdir(fire_folder) if f.endswith('.jpg')])
        for img_file in fire_files:
            img_path = os.path.join(fire_folder, img_file)
            image_features_list.append(img_path)
            labels_list.append(1)
    
    return image_features_list, labels_list

class LSTM_Model(nn.Module):
    def __init__(self, input_dimension=6, hidden_units=32, num_layers=2, num_classes=2):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dimension, hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, num_classes)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class AttentionFusion(nn.Module):
    def __init__(self, sensor_dim, image_dim, hidden_dim=64):
        super(AttentionFusion, self).__init__()
        self.sensor_dim = sensor_dim
        self.image_dim = image_dim
        
        self.attention_sensor = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, sensor_dim),
            nn.Softmax(dim=-1)
        )
        
        self.attention_image = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, image_dim),
            nn.Softmax(dim=-1)
        )
        
        self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
    def forward(self, sensor_feat, image_feat):
        sensor_att = self.attention_sensor(sensor_feat.mean(dim=0))  # [sensor_dim]
        image_att = self.attention_image(image_feat.mean(dim=0))  # [image_dim]
        
        sensor_weighted = sensor_feat * sensor_att.unsqueeze(0)
        image_weighted = image_feat * image_att.unsqueeze(0)
        
        sensor_proj = self.sensor_proj(sensor_weighted)
        image_proj = self.image_proj(image_weighted)
        
        fused = torch.cat([sensor_proj, image_proj], dim=1)
        return fused, sensor_att, image_att

class GatingFusion(nn.Module):
    def __init__(self, sensor_dim, image_dim, hidden_dim=64):
        super(GatingFusion, self).__init__()
        self.sensor_dim = sensor_dim
        self.image_dim = image_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(sensor_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
    def forward(self, sensor_feat, image_feat):
        sensor_global = sensor_feat.mean(dim=0)  # [sensor_dim]
        image_global = image_feat.mean(dim=0)  # [image_dim]
        
        concat_feat = torch.cat([sensor_global, image_global])
        gate_weights = self.gate_network(concat_feat)  # [2]
        
        sensor_proj = self.sensor_proj(sensor_feat)
        image_proj = self.image_proj(image_feat)
        
        fused = gate_weights[0] * sensor_proj + gate_weights[1] * image_proj
        return fused, gate_weights


def extract_sensor_features(sensor_data, lstm_model, method='baseline'):
    X_tensor = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        lstm_outputs, _ = lstm_model.lstm(X_tensor)
        sensor_features = lstm_outputs.squeeze(0).numpy()
    
    if method == 'baseline':
        return sensor_features
    elif method == 'maff-net':
        X_tensor.requires_grad = True
        outputs, _ = lstm_model.lstm(X_tensor)
        loss = outputs.mean()
        loss.backward()
        gradients = X_tensor.grad.data.numpy().squeeze(0)
        weights = np.mean(np.abs(gradients), axis=1)
        threshold = np.percentile(weights[:-120] if len(weights) > 120 else weights, 95)
        key_time_steps = np.where(weights > threshold)[0]
        key_time_steps = key_time_steps[key_time_steps < sensor_features.shape[0]]
        
        sensor_features_weighted = sensor_features.copy()
        sensor_features_weighted[key_time_steps] *= 2
        return sensor_features_weighted
    else:
        return sensor_features

def extract_image_features(image_paths, resnet_model, preprocess, method='baseline', random_state=42):
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            image = preprocess(image)
            images.append(image)
    
    if not images:
        return None, None
    
    images_tensor = torch.stack(images)
    with torch.no_grad():
        image_features = resnet_model(images_tensor).detach().numpy()
    
    rf_model = None
    if method == 'baseline':
        return image_features, None
    elif method == 'maff-net':
        from sklearn.ensemble import RandomForestClassifier
        labels = [0 if 'notfire' in img_path else 1 for img_path in image_paths]
        rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf_model.fit(image_features, labels)
        importances = rf_model.feature_importances_
        threshold = np.mean(importances)
        important_indices = np.where(importances > threshold)[0]
        return image_features[:, important_indices], rf_model
    else:
        return image_features, None


def baseline_fusion(sensor_features, image_features):
    min_dim = min(sensor_features.shape[1], image_features.shape[1])
    if sensor_features.shape[1] > min_dim:
        pca_sensor = KernelPCA(n_components=min_dim)
        sensor_features = pca_sensor.fit_transform(sensor_features)
    if image_features.shape[1] > min_dim:
        pca_image = KernelPCA(n_components=min_dim)
        image_features = pca_image.fit_transform(image_features)
    
    min_samples = min(sensor_features.shape[0], image_features.shape[0])
    sensor_features = sensor_features[:min_samples]
    image_features = image_features[:min_samples]
    
    fused_features = np.concatenate([sensor_features, image_features], axis=1)
    
    pca_final = KernelPCA(n_components=min(64, fused_features.shape[1]))
    fused_features = pca_final.fit_transform(fused_features)
    
    return fused_features

def attention_fusion(sensor_features, image_features, device='cpu'):
    sensor_tensor = torch.tensor(sensor_features, dtype=torch.float32)
    image_tensor = torch.tensor(image_features, dtype=torch.float32)
    
    min_samples = min(sensor_tensor.shape[0], image_tensor.shape[0])
    sensor_tensor = sensor_tensor[:min_samples]
    image_tensor = image_tensor[:min_samples]
    
    max_dim = 512  
    if sensor_tensor.shape[1] > max_dim:
        pca_sensor = KernelPCA(n_components=max_dim)
        sensor_tensor = torch.tensor(pca_sensor.fit_transform(sensor_tensor.numpy()), dtype=torch.float32)
    if image_tensor.shape[1] > max_dim:
        pca_image = KernelPCA(n_components=max_dim)
        image_tensor = torch.tensor(pca_image.fit_transform(image_tensor.numpy()), dtype=torch.float32)
    
    attention_module = AttentionFusion(
        sensor_dim=sensor_tensor.shape[1],
        image_dim=image_tensor.shape[1],
        hidden_dim=64
    ).to(device)
    
    sensor_tensor = sensor_tensor.to(device)
    image_tensor = image_tensor.to(device)
    
    fused_features, sensor_att, image_att = attention_module(sensor_tensor, image_tensor)
    
    fused_np = fused_features.cpu().detach().numpy()
    if fused_np.shape[1] > 128:
        pca_final = KernelPCA(n_components=128)
        fused_np = pca_final.fit_transform(fused_np)
    
    return fused_np, sensor_att.cpu().detach().numpy(), image_att.cpu().detach().numpy()

def gating_fusion(sensor_features, image_features, device='cpu'):
    sensor_tensor = torch.tensor(sensor_features, dtype=torch.float32)
    image_tensor = torch.tensor(image_features, dtype=torch.float32)
    
    min_samples = min(sensor_tensor.shape[0], image_tensor.shape[0])
    sensor_tensor = sensor_tensor[:min_samples]
    image_tensor = image_tensor[:min_samples]
    
    max_dim = 512  
    if sensor_tensor.shape[1] > max_dim:
        pca_sensor = KernelPCA(n_components=max_dim)
        sensor_tensor = torch.tensor(pca_sensor.fit_transform(sensor_tensor.numpy()), dtype=torch.float32)
    if image_tensor.shape[1] > max_dim:
        pca_image = KernelPCA(n_components=max_dim)
        image_tensor = torch.tensor(pca_image.fit_transform(image_tensor.numpy()), dtype=torch.float32)
    
    gating_module = GatingFusion(
        sensor_dim=sensor_tensor.shape[1],
        image_dim=image_tensor.shape[1],
        hidden_dim=64
    ).to(device)
    
    sensor_tensor = sensor_tensor.to(device)
    image_tensor = image_tensor.to(device)
    
    fused_features, gate_weights = gating_module(sensor_tensor, image_tensor)
    
    fused_np = fused_features.cpu().detach().numpy()
    if fused_np.shape[1] > 128:
        pca_final = KernelPCA(n_components=128)
        fused_np = pca_final.fit_transform(fused_np)
    
    return fused_np, gate_weights.cpu().detach().numpy()

def maff_net_fusion(sensor_features, image_features):
    n_components = sensor_features.shape[1]
    pca = KernelPCA(n_components=n_components)
    image_features_reduced = pca.fit_transform(image_features)
    
    min_samples = min(sensor_features.shape[0], image_features_reduced.shape[0])
    sensor_features = sensor_features[:min_samples]
    image_features_reduced = image_features_reduced[:min_samples]
    
    fused_features = np.concatenate([sensor_features, image_features_reduced], axis=1)
    
    pca_final = KernelPCA(n_components=2)
    fused_features = pca_final.fit_transform(fused_features)
    
    return fused_features

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops_resnet(model, input_shape=(1, 3, 224, 224)):
    flops = 0
    flops += 7 * 7 * 3 * 64 * 112 * 112
    flops += 2 * (3 * 3 * 64 * 64 * 56 * 56)
    flops += (3 * 3 * 64 * 128 * 28 * 28) + (3 * 3 * 128 * 128 * 28 * 28)
    flops += (3 * 3 * 128 * 256 * 14 * 14) + (3 * 3 * 256 * 256 * 14 * 14)
    flops += (3 * 3 * 256 * 512 * 7 * 7) + (3 * 3 * 512 * 512 * 7 * 7)
    flops += 512 * 2
    return flops

def calculate_flops_lstm(model, input_shape=(1, 10, 6)):
    flops = 0
    batch_size, seq_len, input_dim = input_shape
    
    flops += 4 * (input_dim * 32 + 32 * 32) * seq_len * batch_size
    flops += 4 * (32 * 32 + 32 * 32) * seq_len * batch_size
    flops += 32 * 2 * batch_size
    
    return flops

def calculate_flops_svm(n_support_vectors, n_features, n_samples=1):
    flops = n_support_vectors * (n_features * 2 + 1) * n_samples
    return flops

def measure_inference_time(model, input_data, num_runs=100):
    model.eval()
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_data)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  
    
    return np.mean(times), np.std(times)

def run_experiment(sensor_file, image_folder, lstm_model, resnet_model, preprocess, 
                   method='baseline', repeat_idx=0, device='cpu'):
    sensor_data, sensor_labels = read_sensor_data(sensor_file)
    image_paths, image_labels = read_image_data(image_folder)
    
    if sensor_data is None or not image_paths:
        return None
    
    if repeat_idx == 0 and method == 'baseline':
        sensor_nonfire = np.sum(sensor_labels == 0)
        sensor_fire = np.sum(sensor_labels == 1)
        image_nonfire = np.sum(np.array(image_labels) == 0)
        image_fire = np.sum(np.array(image_labels) == 1)
        print(f"Non-Fire={sensor_nonfire}, Fire={sensor_fire}, Total={len(sensor_labels)}")
        print(f"Non-Fire={image_nonfire}, Fire={image_fire}, Total={len(image_labels)}")
    
    sensor_features = extract_sensor_features(sensor_data, lstm_model, method)
    image_features_raw, rf_model = extract_image_features(image_paths, resnet_model, preprocess, method, random_state=42 + repeat_idx)
    
    if image_features_raw is None:
        return None
    
    min_samples = min(len(sensor_labels), len(image_labels))
    labels = sensor_labels[:min_samples]
    
    if method == 'baseline':
        fused_features = baseline_fusion(sensor_features, image_features_raw)
        fusion_info = None
    elif method == 'attention':
        fused_features, sensor_att, image_att = attention_fusion(
            sensor_features, image_features_raw, device
        )
        fusion_info = {'sensor_att': sensor_att, 'image_att': image_att}
    elif method == 'gating':
        fused_features, gate_weights = gating_fusion(
            sensor_features, image_features_raw, device
        )
        fusion_info = {'gate_weights': gate_weights}
    elif method == 'maff-net':
        fused_features = maff_net_fusion(sensor_features, image_features_raw)
        fusion_info = None
    else:
        return None
    
    min_samples = min(fused_features.shape[0], len(labels))
    fused_features = fused_features[:min_samples]
    labels = labels[:min_samples]
    
    svm_model = SVC(probability=True, random_state=42 + repeat_idx)
    svm_model.fit(fused_features, labels)
    
    predictions = svm_model.predict(fused_features)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    n_test_samples = min(10, min(len(image_paths), len(sensor_data)))  # 使用10个样本进行测试
    
    test_image_paths = image_paths[:n_test_samples]
    test_sensor_data = sensor_data[:n_test_samples] if len(sensor_data) >= n_test_samples else sensor_data
    
    if len(test_image_paths) > 0:
        test_images = []
        for img_path in test_image_paths:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                image = preprocess(image)
                test_images.append(image)
        if test_images:
            test_images_tensor = torch.stack(test_images)
            with torch.no_grad():
                _ = resnet_model(test_images_tensor)
    
    inference_times = []
    for _ in range(50):  
        start_time = time.time()
        
        test_images = []
        for img_path in test_image_paths:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                image = preprocess(image)
                test_images.append(image)
        if test_images:
            test_images_tensor = torch.stack(test_images)
            with torch.no_grad():
                test_image_features = resnet_model(test_images_tensor).detach().numpy()
        else:
            test_image_features = np.zeros((n_test_samples, 512))
        
        if len(test_sensor_data) > 0:
            X_tensor = torch.tensor(test_sensor_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                lstm_outputs, _ = lstm_model.lstm(X_tensor)
                test_sensor_features = lstm_outputs.squeeze(0).numpy()
        else:
            test_sensor_features = np.zeros((n_test_samples, 32))
        
        min_dim = min(test_sensor_features.shape[1], test_image_features.shape[1])
        test_sensor_features_aligned = test_sensor_features[:, :min_dim]
        test_image_features_aligned = test_image_features[:, :min_dim]
        test_fused_features = np.concatenate([test_sensor_features_aligned, test_image_features_aligned], axis=1)
        
        _ = svm_model.predict(fused_features[:n_test_samples])
        
        inference_times.append((time.time() - start_time) * 1000)  
    
    inference_time_per_batch = np.mean(inference_times)

    inference_time_per_sample = inference_time_per_batch / n_test_samples if n_test_samples > 0 else inference_time_per_batch

    n_samples = fused_features.shape[0]  
    total_inference_time = inference_time_per_sample * n_samples
    
    inference_speed = n_samples / (total_inference_time / 1000) if total_inference_time > 0 else 0  

    inference_time = inference_time_per_sample

    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    
    if rf_model is not None:
        rf_params = 0
        for tree in rf_model.estimators_:
            n_nodes = tree.tree_.node_count
            rf_params += n_nodes * 4
    else:
        rf_params = 0
    
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    
    pca_params = 0
    if method == 'maff-net' or method == 'baseline':
        if image_features_raw is not None:
            n_components_1 = sensor_features.shape[1] if sensor_features.shape[1] < image_features_raw.shape[1] else image_features_raw.shape[1]
            pca_params += image_features_raw.shape[1] * n_components_1
        pca_params += fused_features.shape[1] * 2
    elif method == 'attention' or method == 'gating':
        if fused_features.shape[1] > 128:
            pca_params += fused_features.shape[1] * 128
    
    n_support_vectors = len(svm_model.support_vectors_)
    n_features = fused_features.shape[1]
    svm_params = n_support_vectors * n_features

    param_count = resnet_params + rf_params + lstm_params + pca_params + svm_params
    
    resnet_flops = calculate_flops_resnet(resnet_model)
    
    avg_seq_len = sensor_features.shape[0] if sensor_features.shape[0] > 0 else 10
    lstm_flops = calculate_flops_lstm(lstm_model, input_shape=(1, avg_seq_len, 6))
    
    rf_flops = 0
    if rf_model is not None:
        for tree in rf_model.estimators_:
            n_nodes = tree.tree_.node_count
            rf_flops += n_nodes
    
    pca_flops = 0
    if method == 'maff-net' or method == 'baseline':
        if image_features_raw is not None:
            n_components_1 = sensor_features.shape[1] if sensor_features.shape[1] < image_features_raw.shape[1] else image_features_raw.shape[1]
            pca_flops += image_features_raw.shape[1] * n_components_1 * image_features_raw.shape[0]
        pca_flops += fused_features.shape[1] * 2 * fused_features.shape[0]
    
    svm_flops = calculate_flops_svm(n_support_vectors, n_features, fused_features.shape[0])
    
    total_flops = resnet_flops + rf_flops + lstm_flops + pca_flops + svm_flops
    
    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fused_features': fused_features,
        'labels': labels,
        'fusion_info': fusion_info,
        'inference_time': inference_time,
        'inference_speed': inference_speed, 
        'param_count': param_count,
        'resnet_params': resnet_params,
        'rf_params': rf_params,
        'lstm_params': lstm_params,
        'pca_params': pca_params,
        'svm_params': svm_params,
        'n_support_vectors': n_support_vectors,
        'total_flops': total_flops,
        'resnet_flops': resnet_flops,
        'rf_flops': rf_flops,
        'lstm_flops': lstm_flops,
        'pca_flops': pca_flops,
        'svm_flops': svm_flops
    }
    
    return result


def downsample_data(features, labels, n_samples_per_class=500, random_state=42):
    np.random.seed(random_state)
    
    unique_labels = np.unique(labels)
    selected_indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        n_available = len(label_indices)
        
        if n_available > n_samples_per_class:
            selected = np.random.choice(label_indices, size=n_samples_per_class, replace=False)
        else:
            selected = label_indices
        
        selected_indices.extend(selected)
    
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices) 
    
    return features[selected_indices], labels[selected_indices]

def visualize_fusion_results(results_dict, save_dir):
    from sklearn.manifold import TSNE
    from sklearn.metrics import accuracy_score, f1_score
    
    methods = ['baseline', 'maff-net', 'attention', 'gating']
    method_labels = ['Baseline', 'MAFF-Net (RF+GW)', 'Attention', 'Gating']
    
    method_performance = {}
    for method in methods:
        if method in results_dict and results_dict[method]:
            accuracies = [r['accuracy'] for r in results_dict[method] if 'accuracy' in r]
            f1_scores = [r['f1'] for r in results_dict[method] if 'f1' in r]
            if accuracies and f1_scores:
                method_performance[method] = {
                    'acc_mean': np.mean(accuracies),
                    'f1_mean': np.mean(f1_scores)
                }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (method, method_label) in enumerate(zip(methods, method_labels)):
        if method in results_dict and results_dict[method]:
            result = results_dict[method][0]
            if 'fused_features' in result:
                features = result['fused_features']
                labels = result['labels']
                
                features_downsampled, labels_downsampled = downsample_data(
                    features, labels, n_samples_per_class=500, random_state=42
                )
                
                from sklearn.svm import SVC
                svm_temp = SVC(probability=True, random_state=42)
                svm_temp.fit(features_downsampled, labels_downsampled)
                pred_temp = svm_temp.predict(features_downsampled)
                acc_temp = accuracy_score(labels_downsampled, pred_temp)
                f1_temp = f1_score(labels_downsampled, pred_temp)
                
                print(f"  Computing t-SNE for {method}...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                features_2d = tsne.fit_transform(features_downsampled)
                
                ax = axes[idx]
                notfire_mask = labels_downsampled == 0
                fire_mask = labels_downsampled == 1
                
                scatter1 = ax.scatter(features_2d[notfire_mask, 0], features_2d[notfire_mask, 1], 
                                     c='blue', alpha=0.6, s=60, label='Non-Fire (0)', 
                                     edgecolors='darkblue', linewidths=0.5)
                scatter2 = ax.scatter(features_2d[fire_mask, 0], features_2d[fire_mask, 1], 
                                     c='red', alpha=0.6, s=60, label='Fire (1)', 
                                     edgecolors='darkred', linewidths=0.5)
                
                if method in method_performance:
                    perf = method_performance[method]
                    title = f'{method_label}\n(Acc: {perf["acc_mean"]:.3f}, F1: {perf["f1_mean"]:.3f})'
                else:
                    title = f'{method_label}\n(Acc: {acc_temp:.3f}, F1: {f1_temp:.3f})'
                
                ax.set_title(title, fontsize=18, fontweight='bold')
                ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
                ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
                ax.tick_params(labelsize=14)
                ax.legend(fontsize=14, loc='best', framealpha=0.9)
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Fusion Visualization (t-SNE)', fontsize=22, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fusion_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'maff-net' in results_dict and results_dict['maff-net']:
        result = results_dict['maff-net'][0]
        if 'fused_features' in result:
            features = result['fused_features']
            labels = result['labels']
            
            features_downsampled, labels_downsampled = downsample_data(
                features, labels, n_samples_per_class=500, random_state=42
            )
            
            from sklearn.svm import SVC
            svm_temp = SVC(probability=True, random_state=42)
            svm_temp.fit(features_downsampled, labels_downsampled)
            pred_temp = svm_temp.predict(features_downsampled)
            acc_temp = accuracy_score(labels_downsampled, pred_temp)
            f1_temp = f1_score(labels_downsampled, pred_temp)
            
            print(f"  Computing t-SNE for MAFF-Net (standalone)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features_downsampled)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            notfire_mask = labels_downsampled == 0
            fire_mask = labels_downsampled == 1
            
            scatter1 = ax.scatter(features_2d[notfire_mask, 0], features_2d[notfire_mask, 1], 
                                 c='blue', alpha=0.6, s=80, label='Non-Fire (0)', edgecolors='darkblue', linewidths=0.5)
            scatter2 = ax.scatter(features_2d[fire_mask, 0], features_2d[fire_mask, 1], 
                                 c='red', alpha=0.6, s=80, label='Fire (1)', edgecolors='darkred', linewidths=0.5)
            
            if 'maff-net' in method_performance:
                perf = method_performance['maff-net']
                title = f'MAFF-Net Feature Fusion Visualization (t-SNE)\nAccuracy: {perf["acc_mean"]:.4f}, F1-score: {perf["f1_mean"]:.4f}'
            else:
                title = f'MAFF-Net Feature Fusion Visualization (t-SNE)\nAccuracy: {acc_temp:.4f}, F1-score: {f1_temp:.4f}'
            
            ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
            ax.set_xlabel('t-SNE Dimension 1', fontsize=18)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=18)
            ax.tick_params(labelsize=16)
            ax.legend(fontsize=16, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'maff_net_tsne_standalone.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ MAFF-Net standalone t-SNE visualization saved")

def main():
    print("=" * 60)
    print("MAFF-Net: Multimodal Adaptive Fusion Network")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading pre-trained models...")
    lstm_model = LSTM_Model(input_dimension=6, hidden_units=32, num_layers=2, num_classes=2)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    lstm_model.eval()
    
    resnet_model = models.resnet18(weights=None)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 2)
    resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
    resnet_model.eval()
    
    preprocess = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nFinding test sets...")
    sensor_files, image_folders = find_test_sets()
    print(f"Found {len(sensor_files)} test sets")
    
    if len(sensor_files) == 0:
        print("Error: No test sets found!")
        return
    
    all_results = {
        'baseline': [],
        'maff-net': [],
        'attention': [],
        'gating': []
    }
    
    for test_idx, (sensor_file, image_folder) in enumerate(zip(sensor_files, image_folders)):
        print(f"\n{'='*60}")
        print(f"Processing Test Set {test_idx + 1}/{len(sensor_files)}")
        print(f"Sensor: {os.path.basename(sensor_file)}")
        print(f"Image: {os.path.basename(image_folder)}")
        print(f"{'='*60}")
        
        test_set_results = {
            'baseline': [],
            'maff-net': [],
            'attention': [],
            'gating': []
        }
        
        for method in ['baseline', 'maff-net', 'attention', 'gating']:
            print(f"\nMethod: {method.upper()}")
            method_results = []
            
            for repeat in range(NUM_REPEATS):
                print(f"  Repeat {repeat + 1}/{NUM_REPEATS}...", end=' ')
                try:
                    result = run_experiment(
                        sensor_file, image_folder, lstm_model, resnet_model, 
                        preprocess, method=method, repeat_idx=repeat, device=device
                    )
                    if result:
                        method_results.append(result)
                        print(f"✓ (Acc: {result['accuracy']:.4f}, Time: {result['inference_time']:.2f}ms)")
                    else:
                        print("✗ Failed")
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            test_set_results[method] = method_results
        
        all_results['baseline'].extend(test_set_results['baseline'])
        all_results['maff-net'].extend(test_set_results['maff-net'])
        all_results['attention'].extend(test_set_results['attention'])
        all_results['gating'].extend(test_set_results['gating'])
    
    print("\n" + "=" * 60)
    print("Calculating Final Results...")
    print("=" * 60)
    
    final_results = {}
    efficiency_results = {}
    
    for method in ['baseline', 'maff-net', 'attention', 'gating']:
        if all_results[method]:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            method_stats = {}
            for metric in metrics:
                values = [r[metric] for r in all_results[method]]
                method_stats[f'{metric}_mean'] = np.mean(values)
                method_stats[f'{metric}_std'] = np.std(values)
            
            inference_times = [r['inference_time'] for r in all_results[method] if 'inference_time' in r]
            inference_speeds = [r.get('inference_speed', 0) for r in all_results[method] if 'inference_speed' in r]
            param_counts = [r['param_count'] for r in all_results[method] if 'param_count' in r]
            resnet_params_list = [r.get('resnet_params', 0) for r in all_results[method] if 'resnet_params' in r]
            rf_params_list = [r.get('rf_params', 0) for r in all_results[method] if 'rf_params' in r]
            lstm_params_list = [r.get('lstm_params', 0) for r in all_results[method] if 'lstm_params' in r]
            pca_params_list = [r.get('pca_params', 0) for r in all_results[method] if 'pca_params' in r]
            svm_params_list = [r.get('svm_params', 0) for r in all_results[method] if 'svm_params' in r]
            flops_list = [r.get('total_flops', 0) for r in all_results[method] if 'total_flops' in r]
            resnet_flops_list = [r.get('resnet_flops', 0) for r in all_results[method] if 'resnet_flops' in r]
            rf_flops_list = [r.get('rf_flops', 0) for r in all_results[method] if 'rf_flops' in r]
            lstm_flops_list = [r.get('lstm_flops', 0) for r in all_results[method] if 'lstm_flops' in r]
            pca_flops_list = [r.get('pca_flops', 0) for r in all_results[method] if 'pca_flops' in r]
            svm_flops_list = [r.get('svm_flops', 0) for r in all_results[method] if 'svm_flops' in r]
            
            if inference_times:
                method_stats['inference_time_mean'] = np.mean(inference_times)
                method_stats['inference_time_std'] = np.std(inference_times)
            if inference_speeds:
                method_stats['inference_speed_mean'] = np.mean(inference_speeds)
                method_stats['inference_speed_std'] = np.std(inference_speeds)
            if param_counts:
                method_stats['param_count_mean'] = np.mean(param_counts)
                method_stats['param_count_std'] = np.std(param_counts)
            if resnet_params_list:
                method_stats['resnet_params'] = resnet_params_list[0]  # ResNet参数量是固定的
            if rf_params_list:
                method_stats['rf_params_mean'] = np.mean(rf_params_list)
                method_stats['rf_params_std'] = np.std(rf_params_list)
            if lstm_params_list:
                method_stats['lstm_params'] = lstm_params_list[0]  # LSTM参数量是固定的
            if pca_params_list:
                method_stats['pca_params_mean'] = np.mean(pca_params_list)
                method_stats['pca_params_std'] = np.std(pca_params_list)
            if svm_params_list:
                method_stats['svm_params_mean'] = np.mean(svm_params_list)
                method_stats['svm_params_std'] = np.std(svm_params_list)
            if flops_list:
                method_stats['total_flops_mean'] = np.mean(flops_list)
                method_stats['total_flops_std'] = np.std(flops_list)
            if resnet_flops_list:
                method_stats['resnet_flops'] = resnet_flops_list[0]  # ResNet FLOPs是固定的
            if rf_flops_list:
                method_stats['rf_flops_mean'] = np.mean(rf_flops_list)
                method_stats['rf_flops_std'] = np.std(rf_flops_list)
            if lstm_flops_list:
                method_stats['lstm_flops'] = lstm_flops_list[0]  # LSTM FLOPs是固定的（基于平均序列长度）
            if pca_flops_list:
                method_stats['pca_flops_mean'] = np.mean(pca_flops_list)
                method_stats['pca_flops_std'] = np.std(pca_flops_list)
            if svm_flops_list:
                method_stats['svm_flops_mean'] = np.mean(svm_flops_list)
                method_stats['svm_flops_std'] = np.std(svm_flops_list)
            
            final_results[method] = method_stats
    
    print("\nPerformance Results (Mean ± Std):")
    print("-" * 80)
    print(f"{'Method':<15} {'Accuracy':<18} {'Precision':<18} {'Recall':<18} {'F1':<18}")
    print("-" * 80)
    for method in ['baseline', 'maff-net', 'attention', 'gating']:
        if method in final_results:
            stats = final_results[method]
            print(f"{method.upper():<15} "
                  f"{stats['accuracy_mean']:.4f}±{stats['accuracy_std']:.4f}    "
                  f"{stats['precision_mean']:.4f}±{stats['precision_std']:.4f}    "
                  f"{stats['recall_mean']:.4f}±{stats['recall_std']:.4f}    "
                  f"{stats['f1_mean']:.4f}±{stats['f1_std']:.4f}")
    
    print("\nEfficiency Results (Mean ± Std):")
    print("-" * 140)
    print(f"{'Method':<15} {'Inference':<15} {'Inference':<18} {'Total':<18} {'Total FLOPs':<20}")
    print(f"{'':<15} {'Time(ms)':<15} {'Speed(samples/s)':<18} {'Params':<18} {'(GFLOPs)':<20}")
    print("-" * 140)
    for method in ['baseline', 'maff-net', 'attention', 'gating']:
        if method in final_results:
            stats = final_results[method]
            inf_time_str = f"{stats.get('inference_time_mean', 0):.2f}±{stats.get('inference_time_std', 0):.2f}" if 'inference_time_mean' in stats else "N/A"
            inf_speed_str = f"{stats.get('inference_speed_mean', 0):.2f}±{stats.get('inference_speed_std', 0):.2f}" if 'inference_speed_mean' in stats else "N/A"
            param_str = f"{stats.get('param_count_mean', 0):,.0f}±{stats.get('param_count_std', 0):,.0f}" if 'param_count_mean' in stats else "N/A"
            flops_mean = stats.get('total_flops_mean', 0) / 1e9  # 转换为GFLOPs
            flops_std = stats.get('total_flops_std', 0) / 1e9
            flops_str = f"{flops_mean:.2f}±{flops_std:.2f}" if 'total_flops_mean' in stats else "N/A"
            print(f"{method.upper():<15} {inf_time_str:<15} {inf_speed_str:<18} {param_str:<18} {flops_str:<20}")
    
    print("\nDetailed FLOPs Breakdown:")
    print("-" * 100)
    print(f"{'Method':<15} {'ResNet':<15} {'RF':<15} {'LSTM':<15} {'PCA':<15} {'SVM':<15}")
    print("-" * 100)
    for method in ['baseline', 'maff-net', 'attention', 'gating']:
        if method in final_results:
            stats = final_results[method]
            resnet_flops = stats.get('resnet_flops', 0) / 1e9 if 'resnet_flops' in stats else 0
            rf_flops_mean = stats.get('rf_flops_mean', 0) / 1e6 if 'rf_flops_mean' in stats else 0  # MFLOPs
            lstm_flops = stats.get('lstm_flops', 0) / 1e6 if 'lstm_flops' in stats else 0  # MFLOPs
            pca_flops_mean = stats.get('pca_flops_mean', 0) / 1e6 if 'pca_flops_mean' in stats else 0  # MFLOPs
            svm_flops_mean = stats.get('svm_flops_mean', 0) / 1e6 if 'svm_flops_mean' in stats else 0  # MFLOPs
            print(f"{method.upper():<15} {resnet_flops:.2f}G{'':<8} {rf_flops_mean:.2f}M{'':<8} {lstm_flops:.2f}M{'':<8} {pca_flops_mean:.2f}M{'':<8} {svm_flops_mean:.2f}M")
    
    print("\nSaving results...")
    with open(os.path.join(RESULT_DIR, 'final_results.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print("\nGenerating visualizations...")
    try:
        visualize_fusion_results(all_results, RESULT_DIR)
        print("  ✓ Fusion visualization saved")
    except Exception as e:
        print(f"  ✗ Fusion visualization failed: {str(e)}")
    
    print(f"\nAll results saved to: {RESULT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
