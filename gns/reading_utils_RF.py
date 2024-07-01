import json

def read_metadata(data_path: str, file_name: str = "metadata.json"):
    # reads metadata from a json file
    with open(f"{data_path}{file_name}", 'rt') as f:
        metadata = json.loads(f.read())
    return metadata


def flags_to_dict(FLAGS):
    # converts command line flags to a dictionary
    return {name: FLAGS[name].value for name in FLAGS}
