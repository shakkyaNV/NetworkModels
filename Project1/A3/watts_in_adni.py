import networkx as nx, numpy as np, pandas as pd
from collections import defaultdict; from itertools import combinations
import os, sys

## MODEL WIDE VARIABLES
MODULE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_DIR)))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')

