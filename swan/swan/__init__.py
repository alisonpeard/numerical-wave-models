#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Python code for SWAN nearshore wave model.

See Holthuijsen (2007) Waves in Oceanic and Coastal Waters
"""

from .default_vars import *
from .initial_conditions import *
from .relations_in import *
from .relations_out import *
from .numerical import *
from .spatial import *
from .spectral import *
from .visuals import *


__version__ = "0.01"
__author__ = """Alison Peard"""