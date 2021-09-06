#from app.api.models import MovieIn, MovieOut, MovieUpdate
#from app.api.schemas import MovieIn, MovieOut, MovieUpdate
#from app.api.db import Movies
from app.api.models import Movies
from app.api.schemas import MovieIn, MovieOut, MovieUpdate
from sqlalchemy.orm import Session

# without async since sqlalchemy doesn't support await. 
# without using databases, async def will not be used.

def add_movie (db: Session, payload: MovieIn):
    query = Movies.insert().values(**payload.dict())
    return db.execute(query=query)

def get_all_movies(db: Session):
    query = Movies.select()
    return db.execute.fetch_all(query=query)

def get_movie(db: Session, item_id: int):
    query = Movies.select(Movies.c.item_id=item_id)
    return db.fetch_one(query=query)

def delete_movie(db:Session, item_id:int):
    query = Movies.delete().where(Movies.c.item_id==item_id)
    return db.execute(query=query)

def update_movie(db: Session, item_id:int, payload: MovieOut):
    query = (Movies.update().where(Movies.c.item_id=item_id).values(**payload.dict()))
    return db.execute(query=query)