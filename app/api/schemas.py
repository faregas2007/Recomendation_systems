from typing import List
from pydantic import BaseModel

class Movie(BaseModel):
    item_id: int
    rating: int
    title: str
    genres: List[str]

class MovieOut(MovieIn):
    item_id: int

class MovieUpdate(MovieIn):
    item_id: Optional[int]=None
    title: Optional[str]=None