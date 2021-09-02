import os
import random
import json
import torch
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
from recsys import config

def get_data():
    os.chdir(config.dataset_dir)
    movie_data = pd.read_csv('u.data', sep='\t', names = ['user_id', 'item_id', 'rating', 'timestamp'], encoding='latin-1')
    icols = ['movie_id', 'movie_title', 'release_date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Flim-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    title_data = pd.read_csv('u.item', sep='|', names = icols, encoding='latin-1')
    
    title_col = []
    genre_col = []
    url_col = []

    genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Flim-Noir', 'Horror', 'Musical', 'Mystery', 'Sci-Fi', 'Thriller', 'War', 'Western']
    cols = title_data[genres].columns
    genre_cols = title_data[genres].apply(lambda x: x>0).apply(lambda x: list(cols[x.values]), axis=1)
    

    for id in movie_data['item_id']:
        # add title, genres, url
        title_col.append(title_data['movie_title'][id-1])
        genre_col.append(genre_cols[id-1])
        url_col.append(title_data['IMDb URL'][id-1])

    movie_data.insert(4, 'title', title_col)
    movie_data.insert(5, 'genres', genre_col)
    movie_data.insert(6, 'IMDb URL', url_col)
    return movie_data

def load_dict(filepath: str) -> Dict:
  """
  Load JSON data from a URL
  Args:
    uri(str): URL of the data source
  Returns:
    A dictionary with the loaded JSON data
  """
  with open(filepath, 'r') as fp:
    d = json.load(fp)
  return d


def save_dict(
    d: Dict, 
    filepath: str, 
    cls=None, 
    sortkeys: bool=False)->None:
    with open(filepath, 'w') as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seed(seed: int=1234)->None:
  """Set seed for reproducibility

  Args:
    seed (int, optional): number to use as the seed. Default to 1234
  """
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # multi-GPU

def delete_experiment(experiment_name: str):
  """Delete an experiment with anme 'experiment'
  Args:
    experiment_name (str): Name of the experiment.
  """

  client = mlflow.tracking.MlflowClient()
  experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
  client.delete_experiment(experiment_id=experiment_id)


def parse_csv(df: pd.DataFrame):
  """Parsed dataframe into json format

  Args:
    df: dataframe in csv file

  Return:
    json format data for REST api. 
  """
  result = df.to_json(orient='records')
  parsed = json.loads(result)
  return parsed

