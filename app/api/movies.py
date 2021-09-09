from fastapi import FasAPI, HTTPException, Header, APIRouter, Request, Depends
from typing import List

from app.api import schemas, db_manager, db

from sqlalchemy.orm import Session
from recsys import utils

movies = APIRouter()

def get_db():
    sess = db.SessionLocal()
    try:
        yield sess
    finally:
        sess.close()

@movies.get('/', response_model=List[MovieIn])
def index(sess: Session=Depends(get_db)):
    return db_manager.get_movies(sess)

@movies.post('/', status_code=201)
def add_movie(payload: MovieIn, sess: Session=Depends(get_db)):
    movie_id = db_manager.add_movie(payload, sess)
    response = {
        'id': movie_id,
        **payload.dict()
    }

    return response

@movies.put('/{item_id}')
def update_movie(item_id: int, payload: MovieUpdate, sess: Session=Depends(get_db)):
    movie = db_manager.get_movies(item_id, sess)
    if movie is None:
        raise HTTPException(status_code=404, detail='Movie not found')
    update_data = payload.dict(exclude_unset=True)

    updated_movie = MovieUpdate(**update_data)
    return db_manager.update_movie(item_id, updated_movie, sess)

@movies.delete('/{item_id}')
def delete_movie(item_id: int, sess: Session=Depends(get_db)):
    movie = db_manager.get_movies(item_id, sess)
    if not movie:
        raise HTTPException(status_code=404, detail='Movie not found')
    return db_manager.delete_movie(item_id, sess)

