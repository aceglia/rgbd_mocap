import json


def load_json(path="../../processing/test_project_bis.json"):

    with open(path) as json_file:
        return json.load(json_file)


if __name__ == "__main__":
    config = load_json()
