import json


def load_json():
    path = 'multiprocess_handler/test_project.json'
    path = 'test_project.json'

    with open(path) as json_file:
        return json.load(json_file)


config = load_json()
