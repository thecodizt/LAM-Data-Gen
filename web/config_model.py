from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

class WithSampleTable(BaseModel):
    name: str

class Config(BaseModel):
    num_tables: int
    sample_data: bool
    scale_factor: float
    foreign_key: bool
    foreign_key_col: Optional[str] = None
    with_sample_tables: Optional[List[WithSampleTable]] = None