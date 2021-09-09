from fastapi import FastAPI
from app.api.movies import movies
from app.api.db import Base, engine

Base.metadata.create_all(engine)
app = FastAPI()

app.include_router(movies)