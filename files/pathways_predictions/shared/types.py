#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Shared data types

from __future__ import annotations
from shared.base import *
from typing_extensions import TypeAlias
from typing import Any, Optional, TypeVar

import dataclasses
from enum import Enum
from uuid import UUID
import numpy as np

# Suggestions Layer types:

class PostingType(Enum):
    JOB = 1
    COURSE = 2

@dataclasses.dataclass(frozen=True)
class User(object):
    id: UUID
    features: list[float]

@dataclasses.dataclass(frozen=True)
class Posting:
    type: PostingType
    id: UUID
    features: list[float]

@dataclasses.dataclass(frozen=True)
class Review:
    id: UUID
    rating: float
    posting_type: PostingType
    features: list[float]

    def __repr__(self) -> str:
        return f"Review(id={str(self.id)}, posting_type={self.posting_type.name}, rating={self.rating})"

class FilterComparator(Enum):
    EQUAL = "="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"

@dataclasses.dataclass(frozen=True)
class PostingFilter:
    field_index: int
    target: float
    comparator: FilterComparator

    def __repr__(self) -> str:
        return f"PostingFilter(field[{self.field_index}] {self.comparator.value} {self.target})"

# Transport Layer types:

class RequestType(Enum):
    PING = 0
    DISCONNECT = 1
    POSTINGS_CHUNK = 2
    POSTINGS_DONE = 3
    REVIEWS_CHUNK = 4
    REVIEWS_DONE = 5
    SUGGESTIONS = 6

class FailureReason(Enum):
    UNKNOWN = 0
    INVALID_REQUEST = 1
    REQUEST_CANCELLED_BY_SERVER = 2
    NO_RECORDS_PROVIDED = 3
    INSUFFICIENT_TRAINING_DATA = 4
    NO_SUGGESTIONS_AVAILABLE = 5
    NO_POSTINGS_MATCHING_FILTERS = 6
    NO_REACHABLE_PREDICTION_SERVERS = 7

@dataclasses.dataclass(frozen=True)
class Request:
    type: RequestType
    posting_type: Optional[PostingType] = None
    quantity: Optional[int] = None
    user: Optional[User] = None
    #filters: list[PostingFilter] = dataclasses.field(default_factory=list)
    postings: list[Posting] = dataclasses.field(default_factory=list)
    reviews: list[Review] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        if self.type == RequestType.POSTINGS_CHUNK:
            return f"Request(type={self.type.name}, num_postings={len(self.postings)})"
        elif self.type == RequestType.REVIEWS_CHUNK:
            return f"Request(type={self.type.name}, num_reviews={len(self.reviews)})"
        elif self.type in (RequestType.POSTINGS_DONE, RequestType.REVIEWS_DONE) and self.quantity is not None:
            return f"Request(type={self.type.name}, quantity={self.quantity})"
        elif self.type == RequestType.SUGGESTIONS and self.posting_type is not None and self.user is not None:
            return f"Request(type={self.type.name}, posting_type={self.posting_type.name}, user_id={str(self.user.id)}"
        else:
            return f"Request(type={self.type.name})"

@dataclasses.dataclass(frozen=True)
class Response:
    success: bool
    reason: FailureReason = FailureReason.UNKNOWN
    suggestions: list[UUID] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        if not self.success:
            return f"Response(success={self.success}, reason={self.reason.name})"
        elif len(self.suggestions) > 0:
            return f"Response(success={self.success}, num_suggestions={len(self.suggestions)})"
        else:
            return f"Response(success={self.success})"

SERIALIZABLE_TRANSMISSION = TypeVar("SERIALIZABLE_TRANSMISSION", Request, Response)



# NOTE: The following are placeholder categories that would ideally not be hard-coded, but
# rather fetched from a database that can be safely updated as business logic requires.

# Data Layer types:

class Industries(Enum):
    GOVERNMENT = 1
    RESEARCH = 2
    BANKING = 3
    SECURITY = 4
    GAMING = 5
    CLOUD = 6
    HARDWARE = 7

class Regions(Enum):
    CANADA_WEST = 1
    CANADA_CENTRAL = 2
    CANADA_EAST = 3
    USA = 4
    INTERNATIONAL = 5

class Positions(Enum):
    INTERN = 1
    JUNIOR = 2
    MIDLEVEL = 3
    SENIOR = 4
    MANAGER = 5

class EducationLevels(Enum):
    NONE = 1
    HIGHSCHOOL = 2
    CERTIFICATION = 3
    BACHELORS = 4
    DUALBACHELORS = 5
    MASTERS = 6
    DOCTORATE = 7

class SkillTags(Enum):
    AI = 1
    DATABASES = 2
    DISTRIBUTED = 3
    COMPILERS = 4
    GRAPHICS = 5
    NETWORK = 6
    ARCHITECTURE = 7
    COMMUNICATION = 8

@dataclasses.dataclass(frozen=True)
class TrainingResult:
    num_epochs: int
    loss: float
    accuracy: float
