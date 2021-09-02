from fastapi import FastAPI
from app.api.movies import movies
#from app.api.db import metadata, database, engine

#metadata.create_all(engine)

app = FastAPI()

"""
# problem with connection to database in psycopg2, using default port 5432.
@app.on_event('startup')
async def startup():
    await database.connect()

@app.on_event('shutdown')
async def shutdown():
    await database.disconnect()
"""

app.include_router(movies)