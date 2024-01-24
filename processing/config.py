import json


def load_json():
    path = '../processing/test_project.json'
    # path = 'test_project.json'

    with open(path) as json_file:
        return json.load(json_file)


config = load_json()
