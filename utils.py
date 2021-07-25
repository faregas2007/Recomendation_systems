import json
import torch
from typing import Dict, List

import numpy as np
import pandas as pd

def load_dict(filepath: str) -> Dict:
  """
  Load JSON data from a URL
  Args:
    uri(str): URL of the data source
  Returns:
    A dictionary with the loaded JSON data
  """
  with open(filepath, 'r') as fp:
    d = json.loads(fp)
  return d
