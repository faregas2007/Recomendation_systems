from sqlalchemy import Table, Column, Integer, String
from app.api.db import engine, Base, session
from recsys import utils

class Movies(Base):
    __tablename__ = 'Movies'

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer)
    rating = Column(Integer)
    title = Column(String(500))

# intialize database ---> Great Expectation
"""
movie_db = utils.get_data().head().drop(['user_id', 'IMDb URL', 'timestamp'], axis=1)
movie_data = utils.parse_csv(movie_db)

try:
    # check if table has already stored inside db
    movies = session.query(models.Movies).all()
    if movie is None:
        session.bulk_insert_mappings(models.Movie, movie_data)
    # check if session in stored works
    for row in result:
        print('item_id', row.item_id, 'title', row.title)
except:
    session.rollback()
finally:
    session.close()
"""
