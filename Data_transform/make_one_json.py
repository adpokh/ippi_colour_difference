import json



def make_one_json():
    dir_name = "Ciede2000_Lab/"
    ext = ".json"
    dataset_names = ["bfd-c", "bfd-d65", "bfd-m", "leeds", "witt"]
    data = dict()
    data["colour_1"] = []
    data["colour_2"] = []
    data["dist"] = []
    data["label"] = []
    data["ciede2000_dist"] = []
    data["ciede2000_label"] = []
    data["colour_1_lab"] = []
    data["colour_2_lab"] = []
    for filename in dataset_names:
        with open(dir_name + filename + ext) as f:
            json_data = json.load(f)
            data["colour_1"] += json_data["colour_1"]
            data["colour_2"] += json_data["colour_2"]
            data["label"] += json_data["label"]
            data["dist"] += json_data["dist"]
            data["ciede2000_dist"] += json_data["ciede2000_dist"]
            data["ciede2000_label"] += json_data["ciede2000_label"]
            data["colour_1_lab"] += json_data["colour_1_lab"]
            data["colour_2_lab"] += json_data["colour_2_lab"]
    file_write_name = dir_name + "united" + ext
    with open(file_write_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def add_reversed():
    dir_name = "Ciede2000_Lab/"
    ext = ".json"
    filename = "united"
    with open(dir_name + filename + ext) as f:
        data = json.load(f)
    col_1 = data["colour_1"].copy()
    col_2 = data["colour_2"].copy()
    col_1_lab = data["colour_1_lab"].copy()
    col_2_lab = data["colour_2_lab"].copy()
    data["colour_1"] += col_2
    data["colour_2"] += col_1
    data["colour_1_lab"] += col_2_lab
    data["colour_2_lab"] += col_1_lab
    data["dist"] += data["dist"]
    data["label"] += data["label"]
    data["ciede2000_dist"] += data["ciede2000_dist"]
    data["ciede2000_label"] += data["ciede2000_label"]
    file_write_name = dir_name + "united+reversed" + ext
    with open(file_write_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def add_same():
    dir_name = "Ciede2000_Lab/"
    ext = ".json"
    filename = "united+reversed"
    with open(dir_name + filename + ext) as f:
        data = json.load(f)
    col_1 = data["colour_1"].copy()
    col_1_lab = data["colour_1_lab"].copy()
    l = len(col_1)
    data["colour_1"] += col_1
    data["colour_2"] += col_1
    data["colour_1_lab"] += col_1_lab
    data["colour_2_lab"] += col_1_lab
    for i in range(l):
        data["dist"].append(0)
        data["label"].append(0)
        data["ciede2000_dist"].append(0)
        data["ciede2000_label"].append(0)
    file_write_name = dir_name + "united+reversed+same" + ext
    with open(file_write_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    

add_same()