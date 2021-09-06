from sqlalchemy import Column, Integer, MetaData, String, Table, ARRAY
from sqlalchemy import create_engine, MetaData

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from recsys import utils

from app.api.models import Movies

db_uri = "sqlite:///movie.db"
engine = create_engine(db_uri)

# using database
session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()
# initialize data_base
movie_db = utils.get_data().head().drop(['user_id', 'IMDb URL', 'timestamp'], axis=1)
move_data = utils.parse_csv(movie_db)

try:
    # checking if stored tables are inside db 
    movies = session.query(models.Movies).all()
    if not has_idenity(movies):
        session.bulk_insert_mappings(Movies, movie_data)
    
    # check if session stored works, should put in test.
    result = session.query(Movies).all()
    for row in result:
        print('item', row.item_id, 'title', row.title)
except:
    session.rollback()
finally:
    session.expire_all()
    session.close()
    