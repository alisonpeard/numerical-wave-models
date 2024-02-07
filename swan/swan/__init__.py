#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Python code for SWAN nearshore wave model.

References:
-----------
..[1] Holthuijsen (2007) Waves in Oceanic and Coastal Waters
..[2] Scientific and Technical Documentation SWAN Cycle III v41.45, https://swanmodel.sourceforge.io/online_doc/swantech/swantech.html
"""

from .default_vars import *
from .initial_conditions import *
from .relations_in import *
from .relations_out import *
from .numerical import *
from .solvers import *
from .spatial import *
from .spectral import *
from .visuals import *
from .SWAN import *

__version__ = "0.01"
__author__ = """Alison Peard"""