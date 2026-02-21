import json

def add_label(file_read, file_write):
    with open(file_read) as f:
        data = json.load(f)
    labels = []
    for dist in data["dist"]:
        if (dist >= 1):
            labels.append(1)
        else:
            labels.append(0)
    data["label"] = labels
    with open(file_write, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

add_label("Transformed_Data/berns.json", "Labels/berns.json")