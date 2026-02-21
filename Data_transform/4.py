import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import json
import draw
import matplotlib.pyplot as plt
from colour.plotting import plot_chromaticity_diagram_CIE1931
import colour

def draw_lines(ax, x_coords_1, y_coords_1, x_coords_2, y_coords_2, colour='black'):
    ax.plot(
        [x_coords_1, x_coords_2],
        [y_coords_1, y_coords_2],
        '-',
        color=colour,
        linewidth=0.8,
        alpha=0.5,
        zorder=2
    )

def draw_points(ax, x_coords_1, y_coords_1, x_coords_2, y_coords_2, colour_1='red', colour_2='blue'):
    ax.scatter(
        x_coords_1,
        y_coords_1,
        color=colour_1,
        s=20,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8,
        zorder=3  # Чтобы точки были поверх треугольника
    )
    ax.scatter(
        x_coords_2,
        y_coords_2,
        color=colour_2,
        s=10,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8,
        zorder=3  # Чтобы точки были поверх треугольника
    ) 

# centers = [{"x": 0.3867002206669442, "y": 0.42749993390444, "Y": 69.53}, 
#            {"x": 0.3142003967326897, "y": 0.33099994478024664, "Y": 31.17}, 
#            {"x": 0.2510000120665106, "y": 0.3629003064893694, "Y": 24.06}, 
#            {"x": 0.48320080838215307, "y": 0.34260005533568316, "Y": 14.24}, 
#            {"x": 0.2177000799415229, "y": 0.21439974184021962, "Y": 8.77}]


# for i in range(1, 7):
#     for j in range(1, 7):
#         if (i + j <= 10):
#             d = dict()
#             d["x"] = i/10
#             d["y"] = j/10
#             d["Y"] = 30
#             centers.append(d)


# centers = [
#     {"x": 0.258, "y": 0.45, "Y": 30},
#     {"x": 0.441, "y": 0.198, "Y": 30},
#     {"x": 0.28, "y": 0.385, "Y": 30},
#     {"x": 0.212, "y": 0.55, "Y": 30},
#     {"x": 0.15, "y": 0.68, "Y": 30},
#     {"x": 0.51, "y": 0.236, "Y": 30},
#     {"x": 0.38, "y": 0.498, "Y": 30},
#     {"x": 0.16, "y": 0.2, "Y": 30},
#     {"x": 0.39, "y": 0.237, "Y": 30},
#     {"x": 0.385, "y": 0.393, "Y": 30},
#     {"x": 0.344, "y": 0.284, "Y": 30},
#     {"x": 0.27, "y": 0.275, "Y": 30},
#     {"x": 0.228, "y": 0.25, "Y": 30},
#     {"x": 0.152, "y": 0.365, "Y": 30},
#     {"x": 0.187, "y": 0.118, "Y": 30},
#     {"x": 0.253, "y": 0.125, "Y": 30},
#     {"x": 0.16, "y": 0.057, "Y": 30},
#     {"x": 0.365, "y": 0.153, "Y": 30},
#     {"x": 0.527, "y": 0.35, "Y": 30},
#     {"x": 0.305, "y": 0.323, "Y": 30},
#     {"x": 0.596, "y": 0.283, "Y": 30},
#     {"x": 0.131, "y": 0.521, "Y": 30},
#     {"x": 0.278, "y": 0.223, "Y": 30},
#     {"x": 0.3, "y": 0.163, "Y": 30},
#     {"x": 0.472, "y": 0.399, "Y": 30},
#     {"x": 0.475, "y": 0.3, "Y": 30}
# ]

centers = [
    {"x": 0.258, "y": 0.45, "Y": 15},
    {"x": 0.441, "y": 0.198, "Y": 15},
    {"x": 0.28, "y": 0.385, "Y": 15},
    {"x": 0.212, "y": 0.55, "Y": 15},
    {"x": 0.15, "y": 0.68, "Y": 15},
    {"x": 0.51, "y": 0.236, "Y": 15},
    {"x": 0.38, "y": 0.498, "Y": 15},
    {"x": 0.16, "y": 0.2, "Y": 15},
    {"x": 0.39, "y": 0.237, "Y": 15},
    {"x": 0.385, "y": 0.393, "Y": 15},
    {"x": 0.344, "y": 0.284, "Y": 15},
    {"x": 0.27, "y": 0.275, "Y": 15},
    {"x": 0.228, "y": 0.25, "Y": 15},
    {"x": 0.152, "y": 0.365, "Y": 15},
    {"x": 0.187, "y": 0.118, "Y": 15},
    {"x": 0.253, "y": 0.125, "Y": 15},
    {"x": 0.16, "y": 0.057, "Y": 15},
    {"x": 0.365, "y": 0.153, "Y": 15},
    {"x": 0.527, "y": 0.35, "Y": 15},
    {"x": 0.305, "y": 0.323, "Y": 15},
    {"x": 0.596, "y": 0.283, "Y": 15},
    {"x": 0.131, "y": 0.521, "Y": 15},
    {"x": 0.278, "y": 0.223, "Y": 15},
    {"x": 0.3, "y": 0.163, "Y": 15},
    {"x": 0.472, "y": 0.399, "Y": 15},
    {"x": 0.475, "y": 0.3, "Y": 15}
]

class ColourNet(torch.nn.Module):
    def __init__(self):
        super(ColourNet, self).__init__()
        self.fc1 = torch.nn.Linear(27, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model_path = "colour_net_model_lab.pth"
checkpoint = torch.load(model_path, weights_only=False)

model = ColourNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Переводим модель в режим оценки
best_threshold = checkpoint['best_threshold']
poly = checkpoint['poly_features']

def xyY_to_Lab(x, y, Y, illuminant=np.array([94.81, 100, 107.33])):
    xyY = np.array([x, y, Y])
    XYZ = colour.xyY_to_XYZ(xyY)  # xyY → XYZ
    Lab = colour.XYZ_to_Lab(XYZ, illuminant)  # XYZ → Lab
    return Lab

def predict_color_match(color1, color2, threshold=best_threshold):
    Lab_1 = xyY_to_Lab(color1['x'], color1['y'], color1['Y'])
    Lab_2 = xyY_to_Lab(color2['x'], color2['y'], color2['Y'])
    features = np.array([[Lab_1[0], Lab_1[1], Lab_1[2], 
                         Lab_2[0], Lab_2[1], Lab_2[2]]])
    features_poly = poly.transform(features)
    features_tensor = torch.tensor(features_poly, dtype=torch.float32)
    with torch.no_grad():
        output = model(features_tensor)
        probability = output.item()
        prediction = 1 if probability > threshold else 0
    
    return prediction

dir_name = "Ciede2000_Lab/"
ext = ".json"
dataset_names = ["united+reversed+same"]


data = dict()
data["colour_1"] = []
data["colour_2"] = []
data["label"] = []
data["ciede2000_label"] = []
for filename in dataset_names:
    with open(dir_name + filename + ext) as f:
        json_data = json.load(f)
        data["colour_1"] += json_data["colour_1"]
        data["colour_2"] += json_data["colour_2"]
        data["label"] += json_data["label"]
        data["ciede2000_label"] += json_data["ciede2000_label"]

# net_correct = 0
# ciede2000_correct = 0
# for i in range(len(data["colour_1"])):
#     prediction = predict_color_match(data["colour_1"][i], data["colour_2"][i])
#     net_correct += (prediction == data["label"][i])
#     ciede2000_correct += (data["ciede2000_label"][i] == data["label"][i])
# print("Точность нейросети:", net_correct / len(data["colour_1"]))
# print("Точность ciede2000:", ciede2000_correct / len(data["colour_1"]))

directions = [{"x": 1, "y": 0, "Y": 0}, 
              {"x": 1, "y": 1, "Y": 0},
              {"x": 0, "y": 1, "Y": 0},
              {"x": -1, "y": 1, "Y": 0},
              {"x": -1, "y": 0, "Y": 0}, 
              {"x": -1, "y": -1, "Y": 0},
              {"x": 0, "y": -1, "Y": 0},
              {"x": 1, "y": -1, "Y": 0},

              {"x": 1, "y": 2, "Y": 0}, 
              {"x": 2, "y": 1, "Y": 0},
              {"x": -1, "y": 2, "Y": 0},
              {"x": -2, "y": 1, "Y": 0},
              {"x": -1, "y": -2, "Y": 0}, 
              {"x": -2, "y": -1, "Y": 0},
              {"x": 1, "y": -2, "Y": 0},
              {"x": 2, "y": -1, "Y": 0}]

prec = 0.001
def return_elipse_net_data(col):
    answer = []
    for direction in directions:
        col_2 = col.copy()
        print(direction)
        count = 0
        while (predict_color_match(col, col_2) == 0 and count < 5000):
            col_2["x"] += prec * direction["x"]
            col_2["y"] += prec * direction["y"]
            col_2["Y"] += prec * direction["Y"]
            count += 1
            # print(col_2)
        answer.append(col_2)
    return answer

ellipses_centers = []
for col in centers:
    ellipses_centers.append(return_elipse_net_data(col))
print(ellipses_centers)
data = dict()
data["colour_1"] = []
data["colour_2"] = []
for i in range(len(ellipses_centers)): 
    for j in range(len(ellipses_centers[i])):
        data["colour_1"].append(centers[i])
        data["colour_2"].append(ellipses_centers[i][j])

fig, ax = plt.subplots(figsize=(8, 6))
plot_chromaticity_diagram_CIE1931(show=False, axes=ax)

x_coords_1 = [point['x'] for point in data['colour_1']]
y_coords_1 = [point['y'] for point in data['colour_1']]
x_coords_2 = [point['x'] for point in data['colour_2']]
y_coords_2 = [point['y'] for point in data['colour_2']]
draw_points(ax, x_coords_1, y_coords_1, x_coords_2, y_coords_2)
draw_lines(ax, x_coords_1, y_coords_1, x_coords_2, y_coords_2)



plt.tight_layout()
plt.show()