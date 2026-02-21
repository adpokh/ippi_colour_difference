import colour
import load_data

def xyY_to_dict(x, y, Y):
    colour_res = dict()
    colour_res["x"] = x
    colour_res["y"] = y
    colour_res["Y"] = Y
    return colour_res

def xyz_to_xyY(colour_xyz):
    colour_xyY = colour.XYZ_to_xyY(colour_xyz)
    return xyY_to_dict(colour_xyY[0], colour_xyY[1], colour_xyY[2])

def Lab_to_xyY(L, a, b):
    lab = np.array([L, a, b])
    xyz = colour.Lab_to_XYZ(lab)
    return xyz_to_xyY(xyz)

def xyY_to_Lab(x, y, Y, illuminant):
    xyY = np.array([x, y, Y])
    XYZ = colour.xyY_to_XYZ(xyY)
    Lab = colour.XYZ_to_Lab(XYZ, illuminant)
    return Lab

def transform_dv_pair_xyz(file_read_name, file_write_name):
    data = read_data(file_read_name)
    d = dict()
    colours_1 = []
    colours_2 = []
    for pair in data["pairs"]:
        colour_1 = xyz_to_xyY(data["xyz"][pair[0]])
        colour_2 = xyz_to_xyY(data["xyz"][pair[1]])
        colours_1.append(colour_1)
        colours_2.append(colour_2)
    d["colour_1"] = colours_1
    d["colour_2"] = colours_2
    d["dist"] = data["dv"]
    write_data(file_write_name, d)

def add_label(file_read, file_write):
    data = read_data(file_read_name)
    labels = []
    for dist in data["dist"]:
        if (dist >= 1):
            labels.append(1)
        else:
            labels.append(0)
    data["label"] = labels
    write_data(file_write_name, data)

