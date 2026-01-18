import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax

# 设置文件路径
data_path = r"I:\新大空间视频\训练传感器数据\2024-01-24_11-49-53_1_0.75_3.xlsx"

class MyModel(nn.Module):
    def __init__(self, input_dimension, hidden_units, num_layers, num_classes):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_dimension, hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def train_model(model, epochs, optimizer, data, labels):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    return model  

input_dimension = 6  
hidden_units = 32  
num_layers = 2  
num_classes = 2  

criterion = nn.CrossEntropyLoss()

sheets = ["1m", "2m", "3m", "4m", "5m", "6m"]
for sheet in sheets:
    df = pd.read_excel(data_path, sheet_name=sheet)
    X = df.iloc[:, [2, 4, 6, 8, 10, 12]].values  
    y = df.iloc[:, -1].str.strip().replace({'notfire': 0, 'fire': 1}).values.astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled).float()
    y_tensor = torch.tensor(y).long()

    model = MyModel(input_dimension, hidden_units, num_layers, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(model, 100, optimizer, X_tensor, y_tensor)

    torch.save(trained_model.state_dict(), f"lstm_model_{sheet}.pth")
