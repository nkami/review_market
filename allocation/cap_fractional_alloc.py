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

  agents, objects, agent_prefs, agent_caps, object_caps, min_object_per_reviewer, max_object_per_reviewer = utils.read_cost_file(parsed_args.data_file)
  model, var_assigned, utility = models.max_fractional_model(agents, objects, agent_prefs, agent_caps, object_caps, min_object_per_reviewer, max_object_per_reviewer)
  if parsed_args.verbose: utils.pretty_print_fractional_model(agents, objects, agent_prefs, agent_caps, object_caps, min_object_per_reviewer, max_object_per_reviewer)

  # Quiet Down the Optimizer...
  model.setParam(gpy.GRB.Param.OutputFlag, False )
  model.optimize()
  if model.getAttr('Status') == 2:
  	print("Found Fractional Max Sum Assignment in {} seconds".format(model.Runtime))
  	if parsed_args.verbose: utils.pretty_print_fractional_solution(model, agents, objects, var_assigned, utility)
 #  		#if parsed_args.pickle_file != False:
 #    	#	pk = utils.get_allocation_pickle("Utilitarian SW", agent_caps, agents, object_caps, objects, agent_prefs, var_assigned)
 #    	#	with open(parsed_args.pickle_file + "-utilitarian.pickle", 'wb') as f:
 #      	#		pickle.dump(pk, f, protocol=pickle.HIGHEST_PROTOCOL)
	# else:
 #  		print("Could not find an optimal Utilitarian solution.")











