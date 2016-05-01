""" links dependencies with higher up project directories """
import sys
import os

#insert at beginning of path
sys.path.insert(0, os.path.abspath('./client'))
sys.path.insert(0, os.path.abspath('./server'))
sys.path.insert(0, os.path.abspath('..'))