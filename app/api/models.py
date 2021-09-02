from typing import List
from pydantic import BaseModel

class Movie(BaseModel):
    item_id: int
    rating: int
    title: str
    genres: List[str]