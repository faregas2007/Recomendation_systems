from app.api import models
from app.api import schemas
from sqlalchemy.orm import Session
from recsys import utils

def add_movie(payload: schemas.MovieIn, db: Session):
    row = models.Movies(**payload.dict())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

def get_movies(db: Session):
    query = db.query(models.Movies).all()
    return utils.to_array(query)

def get_movie(item_id: int, db: Session):
    query = db.query(models.Movies).filter(models.Movies.item_id == item_id).first()
    return query.asdict(excude=['id'])

def delete_movie(item_id: int, db: Session):
    query = db.query(models.Movies).filter(models.Movies.item_id == item_id).first()
    return query.asdict(excude=['id'])

def update_movie(item_id: int, payload: schemas.MovieUpdate, db:Session):
    query = db.query(model.Movies).filter(models.Movies.item_id == item_id).first()
    query.ratings = payload.rating
    query.title = payload.title
    db.commit()
    db.refresh(query)
    return query.asdict(excude=['id'])
