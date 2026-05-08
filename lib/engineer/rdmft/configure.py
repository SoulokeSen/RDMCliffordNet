#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:27:37 2026

@author: souloke
"""
from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration data for the calculation.
    """

    max_memory: int = 7000
    """Maximum memory in MB"""

    verbosity: int = 5
    """Verbosity level"""