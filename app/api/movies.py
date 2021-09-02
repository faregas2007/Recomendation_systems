from fastapi import FastAPI, HTTPException, Header, APIRouter
from pydantic import BaseModel
from typing import List

from app.api.models import Movie
from recsys.utils import get_data, parse_csv
from recsys import utils

movies = APIRouter()
movie_db = utils.get_data().head().drop(['user_id', 'IMDb URL', 'timestamp'], axis=1)

@movies.get('/', response_model=List[Movie])
async def index():
    return utils.parse_csv(movie_db)

@movies.post('/', status_code=201)
async def add_movie(payload: Movie):
    movie = payload.dict()
    movie_db.append(movie)
    return {'item_id': len(movie_db['item_id'])-1}

@movies.put('/{item_id}')
async def update_movie(item_id: int, payload: Movie):
    movie = payload.dict()
    movie_length = len(movie_db['item_id'])
    if 0 <= item_id <= movie_length:
        movie_db[item_id] = movie
        return None
    return HTTPException(status_code=404, detail ="Movie with given item_id not found")

@movies.delete('/{item_id}')
async def delete_movie(item_id: int):
    movies_length = len(movie_db['item_id'])
    if 0 <= item_id <= movies_length:
        del movie_db[item_id]
        return None
    raise HTTPException(status_code=404, detail="Movie with given item_id not found")
