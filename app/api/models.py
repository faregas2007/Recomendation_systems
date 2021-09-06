from sqlalchemy import Table
from app.api.db import Base

class Movies(Base):
    __tablename__ = 'movies',

    id = Column(integer, primary_key=True)
    item_id = Column(Integer)
    title = Column(String)

