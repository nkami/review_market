'''
  File:   build_models.py
  Author: Nicholas Mattei (nsmattei@gmail.com)
  Date: October 12, 2018

  About
  --------------------
    This builds some basic allocation models into an LP.  Requires Gurobi.

    TODO: All of these make the assumption that each agent can only be allocated a particular item one time
    relaxing this assumption requires redefining the variables.

'''
import math
import random
import itertools
import sys
import gurobipy as gpy
import utility_functions as utils


def max_fractional_model(agents, objects, agent_prefs, agent_caps, object_caps, min_object_per_reviewer, max_object_per_reviewer):
  '''
    Build a fractional model where we have mins and maxes as well as multiple capacities for each 
    of the agents.
  '''
  m = gpy.Model('Frac')

  # Dicts to keep track of varibles...
  assigned = {}
  utility = {}
  
  # Create a real valued variable  
  for a in agents:
    for o in objects:
      assigned[a,o] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='assigned_%s_%s' % (a,o))

  # Create a variable for each agent's utility.
  for a in agents:
    utility[a] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='utility_%s' % (a))

  # add the variables to the model.
  m.update()

  # Setup the Constraints...
  # Agents have a (possibly 0) capacity for every paper.
  for a in agents:
    for o in objects:
      m.addConstr(assigned[a,o] <= agent_caps[a][o])

  # Enforce that items can only be allocated o times each..
  for o in objects:
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) >= object_caps[o][0], 'object_min_cap_%s' % (o))
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) <= object_caps[o][1], 'object_max_cap_%s' % (o))

  # Enforce that each agent can't have more than agent_cap items.
  for a in agents:
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) >= min_object_per_reviewer, 'agent_min_cap_%s' % (a))
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) <= max_object_per_reviewer, 'agent_max_cap_%s' % (a))

  # Enforce the agent utility computations.
  for a in agents:
    m.addConstr(gpy.quicksum(assigned[a,o] * agent_prefs[a][o] for o in agent_prefs[a].keys()) == utility[a], 'agent_%s_utility' % (a))

  m.update()

  # Set the objective..
  # Add in that we want to maxamize the sum of the values for each agent.
  m.setObjective(gpy.quicksum(utility[a] for a in agents), gpy.GRB.MAXIMIZE)

  m.update()

  return m, assigned, utility


def build_utility_cap_model(m, agents, objects, agent_prefs, agent_caps, object_caps, par_assignment):

  # Dicts to keep track of varibles...
  assigned = {}
  utility = {}

  #NOTE THAT THESE ARE BINARY SO WE CAN ONLY ASSIGN EACH AGENT ONCE!!
  # Create a binary variable for every agent/object.
  for a in agents:
    for o in objects:
      assigned[a,o] = m.addVar(vtype=gpy.GRB.BINARY, name='assigned_%s_%s' % (a,o))

  # Create a variable for each agent's utility.
  for a in agents:
    utility[a] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='utility_%s' % (a))

  # add the variables to the model.
  m.update()

  # Agents can't be assigned negitive objects (no preference).
  for a in agents:
    for o in objects:
      if agent_prefs[a][o] == -1:
        m.addConstr(assigned[a,o] <= 0)

  # If there is a par assignment, make it so.
  if par_assignment != {}:
    for a in agents:
      for o in objects:
        if par_assignment[a][o] > 0:
          m.addConstr(assigned[a,o] == par_assignment[a][o])

  # Enforce that items can only be allocated o times each..
  for o in objects:
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) >= object_caps[o][0], 'object_min_cap_%s' % (o))
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) <= object_caps[o][1], 'object_max_cap_%s' % (o))

  # Enforce that each agent can't have more than agent_cap items.
  for a in agents:
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) >= agent_caps[a][0], 'agent_min_cap_%s' % (a))
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) <= agent_caps[a][1], 'agent_max_cap_%s' % (a))

  # Enforce the agent utility computations.
  for a in agents:
    m.addConstr(gpy.quicksum(assigned[a,o] * agent_prefs[a][o] for o in agent_prefs[a].keys()) == utility[a], 'agent_%s_utility' % (a))

  m.update()
  return m, assigned, utility

'''
  This uses the default model above and tacks on the utilitarian maximizing objective.
'''

def utilitarian_model(agents, objects, agent_prefs, agent_caps, object_caps, par_assignment = {}):
  m = gpy.Model('Util')

  m, assigned, utility = build_utility_cap_model(m, agents, objects, 
                            agent_prefs, agent_caps, object_caps, par_assignment)


  # Add in that we want to maxamize SW.
  m.setObjective(gpy.quicksum(utility[a] for a in agents), gpy.GRB.MAXIMIZE)

  m.update()
  return m, assigned, utility

'''
  This uses the default model above and tacks on the nash product as a sum of logs.
'''

def nash_model(agents, objects, agent_prefs, agent_caps, object_caps, par_assignment = {}):
  m = gpy.Model('Nash')
  log_agent_prefs = {}
  for a in agent_prefs.keys():
    log_agent_prefs[a] = {k:math.log(float(v)) if v is not 0 else 0 for k,v in agent_prefs[a].items()}

  m, assigned, utility = build_utility_cap_model(m, agents, objects, 
                            log_agent_prefs, agent_caps, object_caps, par_assignment)


  # Add in that we want to maxamize SW.
  m.setObjective(gpy.quicksum(utility[a] for a in agents), gpy.GRB.MAXIMIZE)

  m.update()
  return m, assigned, utility

'''
  This uses the default model above and tacks on the egal-maximizing objective.
'''

def egalitarian_model(agents, objects, agent_prefs, agent_caps, object_caps, par_assignment = {}):
  m = gpy.Model('Egal')

  m, assigned, utility = build_utility_cap_model(m, agents, objects, agent_prefs, 
                                  agent_caps, object_caps, par_assignment)

  # BEGIN EGAL SW
  ##################################################
  # Need to add a constraint to get the right format...
  lower_utility = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='lower_utility')

  m.update()

  for a in agents:
    m.addConstr(utility[a] >= lower_utility, 'agent_%s_lower_utility' % (a))

  # now we want to maxamize lower utility.
  m.setObjective(lower_utility, gpy.GRB.MAXIMIZE)
  ##################################################
  # END EGAL SOCIAL WELFARE

  m.update()
  return m, assigned, utility


def build_ranksum_cap_model(agents, objects, agent_prefs, agent_caps, object_caps, par_assignment = {}):
  '''
    Build a model with indicator variables for the sum over all agents of assignments for a particular rank.

    Data
    -----------


    Notes
    ------------

  '''
  m = gpy.Model('RankMax')

  # Convert the utilities to weak rankings.
  weakrank_map = utils.utilities_to_weakranks(agents, objects, agent_prefs)
  #print("Rankmap: ",weakrank_map)

  # Dicts to keep track of varibles...
  assigned = {}
  rank_signature = {}
  rank_sum = {}

  # Create a binary variable for every agent/object.
  for a in agents:
    for o in objects:
      assigned[a,o] = m.addVar(vtype=gpy.GRB.BINARY, name='assigned_%s_%s' % (a,o))

  #print(assigned)

  # Create a rank vector for every agent (big ass rank matrix)...
  for a in agents:
    for r in weakrank_map[a].keys():
      rank_signature[a,r] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='rank_sig_%s_%s' % (a,r))

  # Keep indicator variables for all values at rank i -- we know all agents have every rank so just take the first...
  for r in weakrank_map[agents[0]]:
    rank_sum[r] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='rank_sum_%s' % (r))

  # add the variables to the model.
  m.update()

  # Agents cannot be assigned unranked items...
  for a in agents:
    for o in set(list(agent_prefs[a].keys())) - set(objects):
       m.addConstr(assigned[a,o] <= 0)

  # If there is a par assignment, make it so.
  if par_assignment != {}:
    for a in agents:
      for o in objects:
        if par_assignment[a][o] > 0:
          m.addConstr(assigned[a,o] == par_assignment[a][o])

  # Enforce that items can only be allocated o times each..
  for o in objects:
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) >= object_caps[o][0], 'object_min_cap_%s' % (o))
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) <= object_caps[o][1], 'object_max_cap_%s' % (o))

  # Enforce that each agent can't have more than agent_cap items.
  for a in agents:
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) >= agent_caps[a][0], 'agent_min_cap_%s' % (a))
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) <= agent_caps[a][1], 'agent_max_cap_%s' % (a))

  # Enforce the agent rank vector computations.
  for a in agents:
    for r in weakrank_map[a].keys():
      # get the indicies for the objects ranked r...
      objects_at_r = weakrank_map[a][r]
      #print("agent, objects",a,objects_at_r)

      m.addConstr(gpy.quicksum(assigned[a,o] for o in objects_at_r ) == rank_signature[a,r], 'agent_%s_rank_%s' % (a,r))

  m.update()

  # Make sure the sum variables are correct...
  for r in rank_sum.keys():
    m.addConstr(gpy.quicksum(rank_signature[a,r] for a in agents) == rank_sum[r], 'rank_sum_link_%s' % (r))

  # We can write the maximization as the weighted sum of the exponential decay where the exponential is the 1/num_objects + 1
  m.update()

  decay_factor = 1.0 / (len(objects) + 5)

  m.setObjective(gpy.quicksum(rank_sum[r] * math.pow(decay_factor, i) for i,r in enumerate(rank_sum.keys())), gpy.GRB.MAXIMIZE)

  m.update()
  #print(rank_sum)
  return m, assigned, rank_signature


def owa_model(agents, objects, agent_prefs, agent_caps, object_caps, owa):
  
  m = gpy.Model('OWA')
  assigned = {}
  utility = {}
  owa_utility = {}

  # Create a binary variable for every agent/object.
  for a in agents:
    for o in objects:
      assigned[a,o] = m.addVar(vtype=gpy.GRB.BINARY, name='assigned_%s_%s' % (a,o))

  # Create a variable for each agent's utility.
  for a in agents:
    utility[a] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='utility_%s' % (a))

  # Create a variable for each agent's OWA utility.
  for a in agents:
    owa_utility[a] = m.addVar(vtype=gpy.GRB.CONTINUOUS, name='owa_utility_%s' % (a))

  # add the variables to the model.
  m.update()

  # Enforce that items can only be allocated o times each..
  for o in objects:
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) >= object_caps[o][0], 'object_min_cap_%s' % (o))
    m.addConstr(gpy.quicksum(assigned[a,o] for a in agents) <= object_caps[o][1], 'object_max_cap_%s' % (o))

  # Enforce that each agent can't have more than agent_cap items.
  for a in agents:
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) >= agent_caps[a][0], 'agent_min_cap_%s' % (a))
    m.addConstr(gpy.quicksum(assigned[a,o] for o in objects) <= agent_caps[a][1], 'agent_max_cap_%s' % (a))

  # Enforce the agent utility computations.
  for a in agents:
    m.addConstr(gpy.quicksum(assigned[a,o] * agent_prefs[a][o] for o in objects) == utility[a], 'agent_%s_utility' % (a))

  m.update()

  # BEGIN OWA SW
  ##################################################
  # Get OWA Welfare...
  # OWA must be the same length as the agent cap...

  positions = range(len(owa))
  # New set of variables for each agent, Xop
  rank = {}
  # Create a binary matrix for every agent for object o at position p
  for a in agents:
    for o in objects:
      for p in positions:
        rank[a,o,p] = m.addVar(vtype=gpy.GRB.BINARY, name='agent_%s_obj_%s_pos_%s' % (a,o,p))
  m.update()

  # Matrix wise constraints that each agent's matrix must obey...
  for a in agents:
    # Objects can have at most one position.
    for o in objects:
      m.addConstr(gpy.quicksum(rank[a,o,p] for p in positions) <= 1.0, 'agent_%s_one_position_%s' % (a,o))
    # Positions can have at most one object
    for p in positions:
      m.addConstr(gpy.quicksum(rank[a,o,p] for o in objects) <= 1.0, 'agent_%s_one_object_%s' % (a,p))
    # Link this with the assignment representation..
    for o in objects:
      m.addConstr(gpy.quicksum(rank[a,o,p] for p in positions) >= assigned[a,o], 'agent_%s_assignment_link_%s' % (a,o))
    # Rank matrix is filled from left to right..
    for p in range(len(owa)-1):
      m.addConstr(gpy.quicksum(rank[a,o,p] for o in objects) >= gpy.quicksum(rank[a,o,p+1] for o in objects))
    # Utility of the underlying objects is decreasing......
    for p in range(len(owa)-1):
      m.addConstr(gpy.quicksum(rank[a,o,p] * agent_prefs[a][o] for o in objects) >= gpy.quicksum(rank[a,o,p+1] * agent_prefs[a][o] for o in objects))

    # Agent OWA Value
    for a in agents:
      m.addConstr(gpy.quicksum(agent_prefs[a][o] * owa[p] * rank[a,o,p] for o,p in itertools.product(objects,positions)) == owa_utility[a], 'owa_utility_agent_%s' % (a))

    m.update()
    m.setObjective(gpy.quicksum(owa_utility[a] for a in agents), gpy.GRB.MAXIMIZE)
  ##################################################
  # END OWA SW
  return m, assigned, utility, owa_utility