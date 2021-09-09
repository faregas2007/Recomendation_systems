from sqlalchemy.ext.declarative import declarative_base
from dictalchemy import DictableModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_uri = "sqlite:///movie.db"
engine = create_engine(db_uri, connect_args={'check_same_thread':False})
Base = declarative_base(cls=DictableModel)

# Using database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()