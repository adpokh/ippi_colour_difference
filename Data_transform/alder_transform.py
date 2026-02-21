import pandas as pd
import numpy as np
import json

df = pd.read_csv(
    'Data/table4.dat',
    sep=r'\s+',          # raw-строка или sep='\\s+'
    comment='#',
    header=None,
    names=['Set', 'x', 'y', 'Y', 'a*10^4', 'a/b', 'theta_deg', 'n']
)


df['a'] = df['a*10^4'] * 1e-4 
df['b'] = df['a'] / df['a/b']


df['theta_rad'] = np.deg2rad(df['theta_deg'])

def get_ellipse_points(row):
    x, y = row['x'], row['y']
    a, b = row['a'], row['b']
    theta = row['theta_rad']
    

    phi_axes = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    phi_45 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    phi_all = np.concatenate([phi_axes, phi_45])
    
    # Исходные точки (коэффициент 1.0)
    x_points = x + a * np.cos(theta) * np.cos(phi_all) - b * np.sin(theta) * np.sin(phi_all)
    y_points = y + a * np.sin(theta) * np.cos(phi_all) + b * np.cos(theta) * np.sin(phi_all)
    
    # Сжатые точки (коэффициент 0.7)
    x_scaled = x + 0.7 * (a * np.cos(theta) * np.cos(phi_all) - b * np.sin(theta) * np.sin(phi_all))
    y_scaled = y + 0.7 * (a * np.sin(theta) * np.cos(phi_all) + b * np.cos(theta) * np.sin(phi_all))
    

    return np.column_stack([
        np.concatenate([x_points, x_scaled]),
        np.concatenate([y_points, y_scaled])
    ])


df['ellipse_points'] = df.apply(get_ellipse_points, axis=1)

d = dict()
colours_1 = []
colours_2 = []
dist = []
for index, row in df.iterrows():
    colour_1 = dict()
    colour_1["x"] = row['x']
    colour_1["y"] = row['y']
    colour_1["Y"] = row['Y']
    for i in range(len(row['ellipse_points'])):
        colour_2 = dict()
        colour_2["x"] = row['ellipse_points'][i][0]
        colour_2["y"] = row['ellipse_points'][i][1]
        colour_2["Y"] = row['Y']
        if (i < 8):
            dist.append(1)
        else:
            dist.append(0)
        colours_1.append(colour_1)
        colours_2.append(colour_2)
d["colour_1"] = colours_1        
d["colour_2"] = colours_2
d["dist"] = dist

file_write_name = "Transformed_Data/alder_d65.json"
with open(file_write_name, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=4)