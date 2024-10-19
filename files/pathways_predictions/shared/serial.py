#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Serialization definitions and functions

from __future__ import annotations
from shared.types import *

import json

class Serializer(object):
    DECODE_ERROR = json.JSONDecodeError

    @classmethod
    def _json_convert(cls, o: object) -> object:
        if isinstance(o, UUID):
            return o.hex
        elif isinstance(o, Enum):
            return o.name
    
    @classmethod
    def _strip_dict(cls, d: dict[str, Any]) -> dict[str, Any]:
        for k, v in list(d.items()):
            if v is None or (isinstance(v, list) and len(v) == 0):
                del d[k]
            if isinstance(v, dict) and len(v) > 0:
                v=cls._strip_dict(v)
        return d
    
    @classmethod
    def _dict_convert(cls, d: dict[str, Any]) -> dict[str, Any]:
        for k, v in list(d.items()):
            if k == "type":
                d[k] = RequestType[v]
            
            elif k == "posting_type":
                d[k] = PostingType[v]
            
            elif k == "quantity":
                pass
            
            elif k == "user":
                d[k] = User(
                    id=UUID(v["id"]),
                    features=v["features"]
                )
            
            elif k == "filters":
                d[k] = [PostingFilter(
                    field_index=d2["field_index"],
                    target=d2["target"],
                    comparator=FilterComparator[d2["comparator"]]
                ) for d2 in list(v)]

            elif k == "postings":
                d[k] = [Posting(
                    type=PostingType[d2["type"]],
                    id=UUID(d2["id"]),
                    features=d2["features"]
                ) for d2 in list(v)]

            elif k == "reviews":
                d[k] = [Review(
                    id=UUID(d2["id"]),
                    features=d2["features"],
                    posting_type=PostingType[d2["posting_type"]],
                    rating=d2["rating"]
                ) for d2 in list(v)]

            elif k == "success":
                pass

            elif k == "reason":
                d[k] = FailureReason[v]

            elif k == "suggestions":
                d[k] = [UUID(x) for x in list(v)]
            
            else:
                log.warning(f"Encountered unexpected field while converting dict to dataclass: {k=}, {v=}")
                del d[k]
        
        return d

    @classmethod
    def serialize(cls, data: object) -> bytes:
        data_dict = cls._strip_dict(dataclasses.asdict(data))
        return json.dumps(data_dict, default=cls._json_convert).encode('utf8')
    
    @classmethod
    def deserialize(cls, data: bytes, as_type: type[SERIALIZABLE_TRANSMISSION]) -> SERIALIZABLE_TRANSMISSION:
        data_dict = json.loads(data.decode('utf8'))
        return as_type(**cls._dict_convert(data_dict))