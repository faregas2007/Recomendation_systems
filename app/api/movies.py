from fastapi import FastAPI, HTTPException, Header, APIRouter, Request
from pydantic import BaseModel
from typing import List

from app.api.schemas import MovieIn, MovieOut, MovieUpdate
from recsys import utils

from sqlalchemy.orm import Session
from .db import session, engine

movies = APIRouter()

@app.middleware('http')
async def db_session_middleware(request: Request, call_next):
    response = Request('Internal server error', status_code=500)
    try:
        request.state.db = session()
        response = await call_next()
    finally:
        request.state_db.close()
    return respone

# dependencies
def get_db(request: Request):
    return request.state.db

@movies.get('/', response_model=List[MovieOut])
def index(db: Session):
    return db_manager.get_all_movies(db)

@movies.post('/', status_code=201)
def add_movie(payload: MovieIn, db: Session=Depends(get_db)):
    movie_id = db_manager.add_movie(db, payload)
    response = {
        'id': movie_id,
        **payload.dict()
    }
    return response

@movies.put('/{item_id}')
def update_movie(item_id: int, payload: MovieIn):
    movie = db_manager.get_movie(db, item_id)
    if not movie:
        raise HTTPException(status_code=404, detail='Movie not found')
    update_data = payload.dict(exclude_unset=True)
    movie_in_db = MovieIn(**movie)

    updated_movie = movie_in_db.copy(update=update_data)
    return db_manager.update_movie(item_id, updated_movie)

@movies.delete('/{item_id}')
def delete_movie(item_id: int):
    movie = db_manager.get_movie(db, item_id)
    if not movie:
        raise HTTPException(status_code=404, detail='Movvie not found')
    return db_manager.delete_movie(db, item_id)