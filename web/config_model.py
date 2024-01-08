"""
This module defines the data types and distributions that can be used for generating data without a sample.

Supported field types:
- IntField: Represents an integer field. It has two optional attributes: `min` and `max`, which define the minimum and maximum values that can be generated for this field. If these attributes are not provided, they default to `None`, which means that there are no constraints on the values that can be generated.
- FloatField: Represents a floating-point field. It has two optional attributes: `min` and `max`, which define the minimum and maximum values that can be generated for this field. If these attributes are not provided, they default to `None`.
- RandomStringField: Represents a string field. It generates a random string of a specified length.
- BoolField: Represents a boolean field. It generates either `True` or `False`.
- DateField: Represents a date field. It generates a date within a specified range.
- CategoryStringField: Represents a categorical string field. It generates a string from a specified list of categories.
- GeoField: Represents a geographical field. It generates a geographical coordinate within a specified range.

Supported distributions:
- UNIFORM: Data is evenly distributed across the range.
- NORMAL: Data follows a normal (Gaussian) distribution.
- RANDOM: Data is randomly generated.

Supported table types:
- CROSS_SECTIONAL: Data is organized into a cross-sectional table.
- TIME_SERIES: Data is organized into a time-series table.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

class Distribution(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    RANDOM = "random"

class TableType(Enum): 
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"

# class IntField(BaseModel):
#     min: Optional[int] = None
#     max: Optional[int] = None

# class FloatField(BaseModel):
#     min: Optional[float] = None
#     max: Optional[float] = None

class RandomStringField(BaseModel):
    length: Optional[int] = None

class CategoryStringField(BaseModel):
    categories: List[str] = None
    count: Optional[int] = None
    prompt: Optional[str] = None

class BoolField(BaseModel):
    true_probability: Optional[float] = None

class DateField(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    format: Optional[str] = None

class GeoField(BaseModel):
    lat_range: Optional[List[float]] = None
    lon_range: Optional[List[float]] = None

class WithoutSampleField(BaseModel):
    name: str
    type: str
    type_params: Optional[Union[RandomStringField, BoolField, DateField, CategoryStringField, GeoField]] = None

class WithoutSampleTable(BaseModel):
    name: str
    num_rows: int
    num_cols: int
    table_type: TableType
    generation_method: str
    fields: List[WithoutSampleField]

class WithSampleTable(BaseModel):
    name: str
    num_rows: int
    num_cols: int
    generation_method: str
    TabelType: TableType

class Config(BaseModel):
    num_tables: int
    sample_data: bool
    without_sample_tables: Optional[List[WithoutSampleTable]] = None
    with_sample_tables: Optional[List[WithSampleTable]] = None