import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colour.plotting import plot_chromaticity_diagram_CIE1931
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from draw import draw_points
from draw import draw_lines
import itertools

dir_name = "Labels/"
ext = ".json"
# dataset_names = ["alder_a", "alder_d65", "bfd-c", "bfd-d65", "bfd-m", "leeds", "macadam_1974", "rit-dupont", "witt"]
# dataset_names = ["alder_a", "alder_d65", "leeds", "rit-dupont"]
dataset_names = ["alder_d65", "rit-dupont", "witt", "leeds", "bfd-d65"]

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

df = pd.DataFrame(data)
df['colour1_x'] = df['colour_1'].apply(lambda x: x["x"])
df['colour1_y'] = df['colour_1'].apply(lambda x: x["y"])
df['colour1_Y'] = df['colour_1'].apply(lambda x: x["Y"])

df['colour2_x'] = df['colour_2'].apply(lambda x: x["x"])
df['colour2_y'] = df['colour_2'].apply(lambda x: x["y"])
df['colour2_Y'] = df['colour_2'].apply(lambda x: x["Y"])

columns = ['colour1_x', 'colour1_y', 'colour1_Y', 'colour2_x', 'colour2_y', 'colour2_Y']

features = ['colour1_x', 'colour1_y', 'colour1_Y', 'colour2_x', 'colour2_y', 'colour2_Y']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

for col1, col2 in itertools.combinations(columns, 2):
    features.append(f'{col1}_mul_{col2}')
    df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]

for col in columns:
    features.append(f'{col}_squared')
    df[f'{col}_squared'] = df[col] ** 2

X = df[features]
y = df['label']
print(features)
# Сохраняем индексы перед разбиением
original_indices = X.index


# Разбиваем данные, сохраняя индексы тестовой выборки
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, original_indices, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Предсказанные:")
print(f"Количество 0: {sum(y_pred == 0)}")
print(f"Количество 1: {sum(y_pred == 1)}")

print("Реальные:")
print(f"Количество 0: {sum(y_test == 0)}")
print(f"Количество 1: {sum(y_test == 1)}")


test_points = df.loc[test_indices]

correct_predictions = (y_test == y_pred)
incorrect_predictions = ~correct_predictions

correct_x1 = test_points.loc[correct_predictions, 'colour1_x'].values
correct_y1 = test_points.loc[correct_predictions, 'colour1_y'].values
correct_x2 = test_points.loc[correct_predictions, 'colour2_x'].values
correct_y2 = test_points.loc[correct_predictions, 'colour2_y'].values

incorrect_x1 = test_points.loc[incorrect_predictions, 'colour1_x'].values
incorrect_y1 = test_points.loc[incorrect_predictions, 'colour1_y'].values
incorrect_x2 = test_points.loc[incorrect_predictions, 'colour2_x'].values
incorrect_y2 = test_points.loc[incorrect_predictions, 'colour2_y'].values

def draw_mistakes():
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_chromaticity_diagram_CIE1931(show=False, axes=ax)

    draw_points(ax, correct_x1, correct_y1, correct_x2, correct_y2)
    draw_points(ax, incorrect_x1, incorrect_y1, incorrect_x2, incorrect_y2)

    draw_lines(ax, correct_x1, correct_y1, correct_x2, correct_y2, 'green')
    draw_lines(ax, incorrect_x1, incorrect_y1, incorrect_x2, incorrect_y2, 'red')

    plt.tight_layout()
    plt.show()

def draw_vis():
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_chromaticity_diagram_CIE1931(show=False, axes=ax)

    draw_points(ax, correct_x1, correct_y1, correct_x2, correct_y2)
    draw_points(ax, incorrect_x1, incorrect_y1, incorrect_x2, incorrect_y2)
    for idx, row in test_points.iterrows():
        if row['label'] == 1:
            draw_lines(ax, row['colour1_x'], row['colour1_y'], row['colour2_x'], row['colour2_y'], 'blue')
        else:
            draw_lines(ax, row['colour1_x'], row['colour1_y'], row['colour2_x'], row['colour2_y'], colour='red')
    plt.tight_layout()
    plt.show()



draw_mistakes()