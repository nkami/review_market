'''
  File:   discrete.py
  Author: Nicholas Mattei (nsmattei@gmail.com)
  Date: June 14, 2016

  About
  --------------------
    This is a simple LP for computing a discrete allocation of
    indivisible objects.

    This must be run with the Gruobi python interpreter and Gurobi installed.

    Assignment Algorithms Implemented:
      * Max Utilitarian SW
      * Max Egal SW

    TODO:
      * Rank Maximal
      * SUM-OWA
      * Recursively Balanced

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
  parser = argparse.ArgumentParser(description='''This is the main driver file for building and running discrete allocation models including SUM-OWA models.

    For now this file runs the utilitarian and egalitarian model on a Preflib Formatted input file of your choice and reports the results to the screen and (optionally) saves a pickle of the result to a destination file.
    ''')

  # File to run...
  parser.add_argument('-d', '--data_file', dest='data_file', type=str, required=True, help='PrefLib formatted file of agent preferences.')
  # Destination File
  parser.add_argument('-p', '--pickle_file', dest='pickle_file', type=str, default=False, help='Path and name of output file.')
  # Verboseness
  parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Print results to screen (be verbose).')

  # Agent Caps
  parser.add_argument('-a', '--agent_min', dest='agent_min', type=int, required=True, help='Per agent minimum assignment.')
  parser.add_argument('-A', '--agent_max', dest='agent_max', type=int, required=True, help='Per agent maximum assignment.')

  # Object Caps
  parser.add_argument('-o', '--object_min', dest='object_min', type=int, required=True, help='Per object minimum assignment.')
  parser.add_argument('-O', '--object_max', dest='object_max', type=int, required=True, help='Per object maximum assignment.')

  # algorithms
  parser.add_argument('-u', '--utilitarian', dest='utilitarian', action='store_true', help='Run with the Max Utilitarian Objective.')
  parser.add_argument('-e', '--egalitarian', dest='egalitarian', action='store_true',help='Run with the Egalitarian Objective.')
  parser.add_argument('-r', '--rank_max', dest='rank_maximal', action='store_true',help='Run with the Rank Maximal Objective.')
  parser.add_argument('-l', '--linearowa', dest='linearowa', action='store_true',help='Run with SUM-OAW with Linear OWA Objective.')
  parser.add_argument('-n', '--nash', dest='nash', action='store_true',help='Run with Max Nash Product Objective.')


  parsed_args = parser.parse_args()

  # Get stuff setup.
  agents, objects, agent_prefs = utils.read_preflib_file(parsed_args.data_file)
  agent_caps = {x:(parsed_args.agent_min, parsed_args.agent_max) for x in agents}
  object_caps = {x:(parsed_args.object_min, parsed_args.object_max) for x in objects}

  # print(agents)
  # print(objects)
  # print(agent_prefs)
  # print(agent_caps)
  # print(object_caps)
  # exit()


  # Compute and display the Utilitarian Assignment
  if parsed_args.utilitarian:
    model, var_assigned, utility = models.utilitarian_model(agents, objects, agent_prefs, agent_caps, object_caps)

    # Quiet down the optimizer...
    model.setParam(gpy.GRB.Param.OutputFlag, False )
    model.optimize()
    if model.getAttr('Status') == 2:
      #print("Found Utilitarian Assignment in {} seconds".format(model.Runtime))
      if parsed_args.verbose: utils.pretty_print_utility_solution(model, agents, objects, var_assigned, utility)
      if parsed_args.pickle_file != False:
        pk = utils.get_allocation_pickle("Utilitarian SW", agent_caps, agents, object_caps, objects, agent_prefs, var_assigned)
        with open(parsed_args.pickle_file + "-utilitarian.pickle", 'wb') as f:
          pickle.dump(pk, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Could not find an optimal Utilitarian solution.")


  # Compute and display the Nash assignment
  if parsed_args.nash:
    model, var_assigned, utility = models.nash_model(agents, objects, agent_prefs, agent_caps, object_caps)

    # Quiet down the optimizer...
    model.setParam(gpy.GRB.Param.OutputFlag, False )
    model.optimize()
    if model.getAttr('Status') == 2:
      #print("Found Nash Product Assignment in {} seconds.".format(model.Runtime))
      if parsed_args.verbose: utils.pretty_print_utility_solution(model, agents, objects, var_assigned, utility)
      if parsed_args.pickle_file != False:
        pk = utils.get_allocation_pickle("Nash SW", agent_caps, agents, object_caps, objects, agent_prefs, var_assigned)
        with open(parsed_args.pickle_file + "-nash.pickle", 'wb') as f:
          pickle.dump(pk, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Could not find an optimal Nash Product Assignment solution.")

  # Compute and display the Rank Maximal
  if parsed_args.rank_maximal:
    model, var_assigned, rank_signature = models.build_ranksum_cap_model(agents, objects, agent_prefs, agent_caps, object_caps)

    # Quiet down the optimizer...
    model.setParam(gpy.GRB.Param.OutputFlag, False )
    model.optimize()
    if model.getAttr('Status') == 2:
      #print("Found Rank Maximal Assignment in {} seconds.".format(model.Runtime))
      #print(rank_signature.keys())
      if parsed_args.verbose: utils.pretty_print_rank_solution(model, agents, objects, var_assigned, rank_signature)

      if parsed_args.pickle_file != False:
        pk = utils.get_allocation_pickle("Rank Maximal", agent_caps, agents, object_caps, objects, agent_prefs, var_assigned)
        with open(parsed_args.pickle_file + "-rank_maximal.pickle", 'wb') as f:
          pickle.dump(pk, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Could not find an optimal Rank Maximal solution.")

  # Compute and display the Egalitarian Solution.
  if parsed_args.egalitarian:
    model, var_assigned, utility = models.egalitarian_model(agents, objects, agent_prefs, agent_caps, object_caps)

    # Quiet down the optimizer...
    model.setParam(gpy.GRB.Param.OutputFlag, False )
    model.optimize()
    if model.getAttr('Status') == 2:
      #print("Found Egalitarian Assignment in {} seconds.".format(model.Runtime))
      if parsed_args.verbose: utils.pretty_print_utility_solution(model, agents, objects, var_assigned, utility)
      if parsed_args.pickle_file != False:
        pk = utils.get_allocation_pickle("Egalitarian SW", agent_caps, agents, object_caps, objects, agent_prefs, var_assigned)
        with open(parsed_args.pickle_file + "-egalitarian.pickle", 'wb') as f:
          pickle.dump(pk, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Could not find an optimal Egalitarian solution.")

  if parsed_args.linearowa:
    # Build linear OWA vector.
    owa_length = max([u for l,u in agent_caps.values()])
    linear_owa = list(np.linspace(0,1,owa_length))
    linear_owa.reverse()
    #print(linear_owa)
    model, var_assigned, utility, linear_owa_utility = models.owa_model(agents, objects, agent_prefs, agent_caps, object_caps, linear_owa)
    model.setParam(gpy.GRB.Param.OutputFlag, False )
    model.optimize()
    if model.getAttr('Status') == 2:
      #print("Found Linear SumOWA Assignment in {} seconds.".format(model.Runtime))
      if parsed_args.verbose: utils.pretty_print_utility_solution(model, agents, objects, var_assigned, utility, linear_owa_utility)
      if parsed_args.pickle_file != False:
        pk = utils.get_allocation_pickle("LinearSumOWA SW", agent_caps, agents, object_caps, objects, agent_prefs, var_assigned)
        with open(parsed_args.pickle_file + "-linearsumowa.pickle", 'wb') as f:
          pickle.dump(pk, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Could not find an optimal Linear SumOWA solution.")

