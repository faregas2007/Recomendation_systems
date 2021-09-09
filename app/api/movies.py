from fastapi import FastAPI, HTTPException, Header, APIRouter, Request, Depends
from typing import List

from app.api import schemas, db_manager
from app.api.db import SessionLocal

from sqlalchemy.orm import Session
from recsys import utils

movies = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@movies.get('/', response_model=List[schemas.MovieIn])
def index(db: Session=Depends(get_db)):
    return db_manager.get_movies(db=db)

@movies.post('/', status_code=201)
def add_movie(payload: schemas.MovieIn, db: Session=Depends(get_db)):
    movie_id = db_manager.add_movie(payload, db)
    response = {
        'id': movie_id,
        **payload.dict()
    }

    return response

@movies.put('/{item_id}')
def update_movie(item_id: int, payload: schemas.MovieUpdate, db: Session=Depends(get_db)):
    movie = db_manager.get_movies(item_id, db)
    if movie is None:
        raise HTTPException(status_code=404, detail='Movie not found')
    update_data = payload.dict(exclude_unset=True)

    updated_movie = schemas.MovieUpdate(**update_data)
    return db_manager.update_movie(item_id, updated_movie, db)

@movies.delete('/{item_id}')
def delete_movie(item_id: int, db: Session=Depends(get_db)):
    movie = db_manager.get_movies(item_id, db)
    if not movie:
        raise HTTPException(status_code=404, detail='Movie not found')
    return db_manager.delete_movie(item_id, db)

