# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for MASt3R
# --------------------------------------------------------
import os
import hashlib
from typing import Dict, List, Union, Tuple


def mkdir_for(f):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    return f


def hash_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def filter_dict_with_routes(data: Dict, routes: List[Union[str, Tuple[str, ...]]]) -> List:
    """

    @param data: input dict
    @param routes: e.g. ["loss", ("pred1, "pts3d", 0)]
    @return: grabs the specified items so in this case data["loss"] merged with data["pred1"]["pts3d"]
    """
    result = []

    for route in routes:
        if isinstance(route, str):  # Top-level key
            if route in data:
                result.append(data[route])
        if isinstance(route, tuple):  # Nested key path
            temp = data
            for tuple_element in route:
                try:
                    temp = temp[tuple_element]
                except (KeyError, IndexError, TypeError):
                    temp = None
            result.append(temp)

    return result