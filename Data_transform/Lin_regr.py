import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import numpy as np


class ColourNet(nn.Module):
    def __init__(self):
        super(ColourNet, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # Увеличили размер слоя
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1)
        x = self.fc3(x)  # Без сигмоиды!
        return x

dir_name = "Labels/"
ext = ".json"
# dataset_names = ["alder_a", "alder_d65", "bfd-c", "bfd-d65", "bfd-m", "leeds", "macadam_1974", "rit-dupont", "witt"]
dataset_names = ["alder_a", "alder_d65", "leeds", "rit-dupont"]

data = dict()
data["colour_1"] = []
data["colour_2"] = []
data["label"] = []
for filename in dataset_names:
    with open(dir_name + filename + ext) as f:
        json_data = json.load(f)
        data["colour_1"] += json_data["colour_1"]
        data["colour_2"] += json_data["colour_2"]
        data["label"] += json_data["label"]

x1 = [color["x"] for color in data["colour_1"]]
y1 = [color["y"] for color in data["colour_1"]]
Y1 = [color["Y"] for color in data["colour_1"]]

x2 = [color["x"] for color in data["colour_2"]]
y2 = [color["y"] for color in data["colour_2"]]
Y2 = [color["Y"] for color in data["colour_2"]]

labels = data["label"]

features = torch.tensor(list(zip(x1, y1, Y1, x2, y2, Y2)), dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Преобразуем в размер [N, 1]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# на всякий случай нормализация данных
mean = X_train.mean(dim=0)
std = X_train.std(dim=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = ColourNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:  # итерация по батчам
        optimizer.zero_grad()  # обнуляем градиенты
        
        outputs = model(inputs)  # forward pass (предсказание)
        loss = criterion(outputs, labels)  # вычисление ошибки
        
        loss.backward()  # backward pass (вычисление градиентов)
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad mean: {param.grad.mean().item():.4f}")
        optimizer.step()  # обновление весов
    
    print(epoch)
    

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("Test_accuracy:", accuracy, "%")
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        for inputs, labels in train_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    print("Train_accuracy:", accuracy, "%")
        

print("Ready")
model.eval()
correct = 0
total = 0

all_predicted = []
all_outputs = []
all_labels = []
with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
    for inputs, labels in train_loader:
        outputs = model(inputs)
        all_outputs += outputs
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_predicted += predicted
        all_labels += labels

accuracy = 100 * correct / total
print(all_predicted)
print(all_outputs)
print("Train_accuracy:", accuracy, "%")

best_accuracy = 0
best_threshold = 0

for threshold in np.arange(0, 1, 0.001):
    predicted = (torch.tensor(all_outputs) > threshold).float()  # Преобразуем в тензор
    correct = (predicted == torch.tensor(all_labels)).sum().item()
    current_accuracy = 100 * correct / total
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_threshold = threshold

print(f"Best_train_accuracy: {best_accuracy:.2f}%, Best_train_threshold: {best_threshold:.3f}")

correct = 0
total = 0

with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > best_threshold).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
accuracy = 100 * correct / total
print("Best_test_accuracy:", accuracy, "%")