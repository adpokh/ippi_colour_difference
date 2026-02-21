import json

def accuracy(data):
    correct = 0
    for i in range(len(data["label"])):
        if (data["label"][i] == data["ciede2000_label"][i]):
            correct += 1
    return correct/len(data["label"])

dir_name = "Ciede2000/"
ext = ".json"
dataset_names = ["bfd-c", "bfd-d65", "bfd-m", "leeds", "witt", "rit-dupont"]
# dataset_names = ["witt"]
for filename in dataset_names:
    data = dict()
    data["colour_1"] = []
    data["colour_2"] = []
    data["label"] = []
    data["ciede2000_label"] = []
    with open(dir_name + filename + ext) as f:
        json_data = json.load(f)
        data["colour_1"] += json_data["colour_1"]
        data["colour_2"] += json_data["colour_2"]
        data["label"] += json_data["label"]
        data["ciede2000_label"] += json_data["ciede2000_label"]
    print(accuracy(data), filename)