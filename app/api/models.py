from sqlalchemy import Table
from app.api.db import metadata

class Movies(Base):
    __tablename__ = 'movies',

    metadata,
    id = Column(integer, primary_key=True)
    item_id = Column(Integer)
    title = Column(String)

