import os
import json

def read_metadata(data_path: str,
                  file_name: str = "metadata.json"):
  """Read metadata of datasets

  Args:
    data_path (str): Path to metadata JSON file
    file_name (str): Name of metadata file

  Returns:
    metadata json object
  """
  with open(os.path.join(data_path, file_name), 'rt') as fp:
      # The previous format of the metadata does not distinguish the purpose of metadata
      metadata = json.loads(fp.read())

  return metadata

def flags_to_dict(FLAGS):
  flags_dict = {}
  for name in FLAGS:
    flag_value = FLAGS[name].value
    flags_dict[name] = flag_value
  return flags_dict
