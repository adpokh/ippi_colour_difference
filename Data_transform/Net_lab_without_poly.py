import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # Импорт TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import json
import numpy as np
import os
from datetime import datetime

# Создаем директорию для логов TensorBoard с временной меткой
log_dir = os.path.join("runs_lab_augm", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)  # Инициализация writer для TensorBoard

class ColourNet(nn.Module):
    def __init__(self):
        super(ColourNet, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

dir_name = "Ciede2000_Lab/"
ext = ".json"
filename = "united+reversed+same"
with open(dir_name + filename + ext) as f:
    data = json.load(f)

L1 = [color["L"] for color in data["colour_1_lab"]]
a1 = [color["a"] for color in data["colour_1_lab"]]
b1 = [color["b"] for color in data["colour_1_lab"]]

L2 = [color["L"] for color in data["colour_2_lab"]]
a2 = [color["a"] for color in data["colour_2_lab"]]
b2 = [color["b"] for color in data["colour_2_lab"]]

labels = data["label"]

features = np.array(list(zip(L1, a1, b1, L2, a2, b2)))

features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = ColourNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    # Записываем loss и accuracy валидации
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

writer.close()

print("Ready")
model.eval()
correct = 0
total = 0

all_predicted = []
all_outputs = []
all_labels = []
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        all_outputs += outputs
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_predicted += predicted
        all_labels += labels

accuracy = 100 * correct / total
print("Train_accuracy:", accuracy, "%")

best_accuracy = 0
best_threshold = 0

for threshold in np.arange(0, 1, 0.001):
    predicted = (torch.tensor(all_outputs) > threshold).float()
    correct = (predicted == torch.tensor(all_labels)).sum().item()
    current_accuracy = 100 * correct / total
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_threshold = threshold

print(f"Best_train_accuracy: {best_accuracy:.2f}%, Best_train_threshold: {best_threshold:.3f}")

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > best_threshold).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
accuracy = 100 * correct / total
print("Best_test_accuracy:", accuracy, "%")

# Сохраняем модель
model_path = "colour_net_model_lab_augm.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_threshold': best_threshold,
    'poly_features': poly
}, model_path)

print(f"Model saved to {model_path}")