import json

def read_data(file_read_name):
    with open(file_read_name) as file:  
        data = json.load(file)
    return data

def write_data(file_write_name, data):
    with open(file_write_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)