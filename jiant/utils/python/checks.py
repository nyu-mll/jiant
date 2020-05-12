from typing import Dict


def dict_equal(dict1: Dict, dict2: Dict) -> bool:
    if not len(dict1) == len(dict2):
        return False
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        if k1 != k2:
            return False
        if v1 != v2:
            return False
    return True
