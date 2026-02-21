import matplotlib.pyplot as plt
from colour.plotting import plot_chromaticity_diagram_CIE1931
import json

def read_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

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

def draw_points_and_lines(ax, data):
    x_coords_1 = [point['x'] for point in data['colour_1']]
    y_coords_1 = [point['y'] for point in data['colour_1']]
    x_coords_2 = [point['x'] for point in data['colour_2']]
    y_coords_2 = [point['y'] for point in data['colour_2']]
    draw_points(ax, x_coords_1, y_coords_1, x_coords_2, y_coords_2)
    
    x_coords_1_vis = [point['x'] for point, label in zip(data['colour_1'], data['label']) if label == 1]
    y_coords_1_vis = [point['y'] for point, label in zip(data['colour_1'], data['label']) if label == 1]
    x_coords_2_vis = [point['x'] for point, label in zip(data['colour_2'], data['label']) if label == 1]
    y_coords_2_vis = [point['y'] for point, label in zip(data['colour_2'], data['label']) if label == 1]
    
    x_coords_1_not_vis = [point['x'] for point, label in zip(data['colour_1'], data['label']) if label == 0]
    y_coords_1_not_vis = [point['y'] for point, label in zip(data['colour_1'], data['label']) if label == 0]
    x_coords_2_not_vis = [point['x'] for point, label in zip(data['colour_2'], data['label']) if label == 0]
    y_coords_2_not_vis = [point['y'] for point, label in zip(data['colour_2'], data['label']) if label == 0]
    
    draw_lines(ax, x_coords_1_vis, y_coords_1_vis, x_coords_2_vis, y_coords_2_vis, 'blue')
    draw_lines(ax, x_coords_1_not_vis, y_coords_1_not_vis, x_coords_2_not_vis, y_coords_2_not_vis, 'red')
    
def draw(dataset_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_chromaticity_diagram_CIE1931(show=False, axes=ax)
    for filename in dataset_names:
        draw_points_and_lines(ax, read_file(dir_name + filename + ext))
    plt.tight_layout()
    plt.show()


dir_name = "Ciede2000/"
ext = ".json"
# dataset_names = ["alder_a", "alder_d65", "bfd-c", "bfd-d65", "bfd-m", "leeds", "macadam_1974", "rit-dupont", "witt"]
dataset_names = ["united"]

# draw(dataset_names)