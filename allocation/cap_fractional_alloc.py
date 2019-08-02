'''
  File:   fractional_cost.py
  Author: Nicholas Mattei (nsmattei@tulane.edu)
  Date: June 14, 2016

  About
  --------------------
    This is a simple LP that reads a cost and capacity file 
    and finds a fractional allocation that minimizes the total cost.

    Note: Eventually this should get folded into the other 
    file.

'''

import pickle
import argparse
import copy
import sys
import random
import itertools
import numpy as np
import gurobipy as gpy

import utility_functions as utils
import build_models as models


if __name__ == "__main__":
  # Parse some basic arguments...
  parser = argparse.ArgumentParser(description='''This is the main driver file for finding Fractional Allocations that Minimize Social Cost.

  	Right now all specifications are handled by the data file and not as input switches.
    ''')

  # File to run...
  parser.add_argument('-d', '--data_file', dest='data_file', type=str, required=True, help='PrefLib formatted file of agent preferences.')
  # Verboseness
  parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Print results to screen (be verbose).')

  parsed_args = parser.parse_args()

  # Get stuff setup.
  print("Parsing File..")
  utils.read_cost_file(parsed_args.data_file)
  '''
  agent_caps = {x:(parsed_args.agent_min, parsed_args.agent_max) for x in agents}
  object_caps = {x:(parsed_args.object_min, parsed_args.object_max) for x in objects}

  if parsed_args.verbose:
  	utils.pretty_print_fractional()
'''



