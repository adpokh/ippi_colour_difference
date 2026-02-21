import json

dir_name = "Labels/"
ext = ".json"
dataset_names = ["bfd-c", "bfd-d65", "bfd-m", "leeds", "witt"]

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

count_0 = labels.count(0)
count_1 = labels.count(1)

print(f"Количество 0: {count_0}")
print(f"Количество 1: {count_1}")
print(count_1 / (count_0 + count_1))