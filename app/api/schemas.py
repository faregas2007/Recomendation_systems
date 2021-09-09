from pydantic import BaseModel
from typing import List, Optional

class MovieIn(BaseModel):
    item_id: int
    rating: int
    title: str

class MovieOut(MovieIn):
    id: int


class MovieUpdate(MovieIn):
    item_id: Optional[int] = None
    rating: Optional[int] = None
    title: Optional[str] = None

