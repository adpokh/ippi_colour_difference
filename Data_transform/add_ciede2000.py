import json
import colour
import numpy as np

def xyY_to_Lab(x, y, Y, illuminant):
    xyY = np.array([x, y, Y])
    XYZ = colour.xyY_to_XYZ(xyY)  # xyY → XYZ
    Lab = colour.XYZ_to_Lab(XYZ, illuminant)  # XYZ → Lab
    return Lab

def add_ciede2000_label(file_read_name, file_write_name, illuminant):
    with open(file_read_name) as file:  
        data = json.load(file)
    ciede2000 = []
    ciede2000_label = []
    colour_1_lab = []
    colour_2_lab = []
    for i in range(len(data["colour_1"])):
        col_1_lab = xyY_to_Lab(data["colour_1"][i]["x"], data["colour_1"][i]["y"], data["colour_1"][i]["Y"], illuminant)
        col_2_lab = xyY_to_Lab(data["colour_2"][i]["x"], data["colour_2"][i]["y"], data["colour_2"][i]["Y"], illuminant)
        col_1_dict_lab = dict()
        col_1_dict_lab["L"] = col_1_lab[0]
        col_1_dict_lab["a"] = col_1_lab[1]
        col_1_dict_lab["b"] = col_1_lab[2]
        col_2_dict_lab = dict()
        col_2_dict_lab["L"] = col_2_lab[0]
        col_2_dict_lab["a"] = col_2_lab[1]
        col_2_dict_lab["b"] = col_2_lab[2]
        colour_1_lab.append(col_1_dict_lab)
        colour_2_lab.append(col_2_dict_lab)
        delta_e = colour.delta_E(col_1_lab, col_2_lab, method="CIE 2000")
        ciede2000.append(delta_e)
        if (delta_e >= 1):
            ciede2000_label.append(1)
        else:
            ciede2000_label.append(0)
    data["ciede2000_dist"] = ciede2000
    data["ciede2000_label"] = ciede2000_label
    data["colour_1_lab"] = colour_1_lab
    data["colour_2_lab"] = colour_2_lab
    with open(file_write_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


dir_name = "Labels/"
ext = ".json"
dataset_names = ["bfd-c", "bfd-d65", "bfd-m", "leeds", "rit-dupont", "witt"]
# dataset_illuminant = [c, d65, m, d65, d65, d65]
d65 = np.array([94.81, 100, 107.33])
c = np.array([98.07, 100, 118.23])
m = np.array([94.65, 100, 103.97])
dir_write_name = "Ciede2000_Lab/"
for name in dataset_names:
    if (name != 'bfd-c' and name != 'bfd-m'):
        add_ciede2000_label(dir_name+name+ext, dir_write_name+name+ext, d65)
        print("d")
    elif (name == 'bfd-c'):
        add_ciede2000_label(dir_name+name+ext, dir_write_name+name+ext, c)
        print("c")
    else:
        print("m")
        add_ciede2000_label(dir_name+name+ext, dir_write_name+name+ext, m)