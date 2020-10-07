# -*- coding: utf-8 -*-
"""This module contains utils for handling data (e.g., validating data)"""
import hashlib


def md5_checksum(filepath: str) -> str:
    """Calculate MD5 checksum hash for a given file.

    Code from example: https://stackoverflow.com/a/3431838/8734015.

    Args:
        filepath: file to calculate MD5 checksum.

    Returns:
        MD5 hash string.

    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
