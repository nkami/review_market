'''
  File:   utility_functions.py
  Author: Nicholas Mattei (nsmattei@gmail.com)
  Date: Oct. 12, 2018

  About
  --------------------
    Utility functions for working with discrete allocations.

'''

import copy

def utilities_to_weakranks(agents, objects, agent_prefs):
  '''
    Take the set of all possible utility values over all agents and make that the rank indicies and then use that to create a map which maps from an object to a rank of that object.

    Data
    --------

    Notes
    --------
  '''

  # Get the set of all possible utility values and sort it.
  #print(agent_prefs)

  # Maybe a better way to do this but lazy...
  unique_ranks = []
  for a in agents:
    for k,v in agent_prefs[a].items():
      unique_ranks.append(v)
  unique_ranks = list(set(unique_ranks))

  #print(unique_ranks)

  # make a lookup table
  lookup = sorted(unique_ranks, reverse=True)
  # go over all the agents and make a rank signature..
  agent_ranks = {a:{} for a in agents}
  for a in agents:
    ranks = {i:[] for i,v in list(enumerate(unique_ranks))}
    for i in objects:
      if i in agent_prefs[a].keys():
        ranks[lookup.index(agent_prefs[a][i])] += [i]
    agent_ranks[a] = ranks

  #print(agent_ranks)
  return agent_ranks

def pretty_print_model(agents, objects, agent_prefs, agent_caps, object_caps):
  # Print the current model..
  print("{:^10}".format("Objects") + "| Min, Max")
  print("{:-^75}".format(""))
  for o in objects:
    c = object_caps[o]
    print("{:^10}".format(str(o)) + "| " + str(c[0]) + ", " + str(c[1]))
  print("{:-^75}".format(""))
  print("{:^10}".format("Agents") + "|" \
        + "{:^10}".format("Min, Max") + "|" \
        + "{:^50}".format("Object(Utility)") + "|")
  print("{:-^75}".format(""))
  for a in sorted(agent_prefs.keys()):
    c = agent_caps[a]
    print("{:^10}".format(str(a)) + "|" \
        + "{:^10}".format(str(c[0]) + "," + str(c[1])) + "| " \
        + ", ".join([str(o) + "(" + str(u) + ")" for o,u in sorted(agent_prefs[a].items(),key=lambda x:x[1], reverse=True)]))
  print("{:-^75}".format(""))

def pretty_print_fractional_model(agents, objects, agent_prefs, agent_caps, object_caps, min_object_per_reviewer, max_object_per_reviewer):
  # Print the current model..
  print("{:^10}".format("Objects") + "| Min, Max")
  print("{:-^75}".format(""))
  for o in objects:
    c = object_caps[o]
    print("{:^10}".format(str(o)) + "| " + str(c[0]) + ", " + str(c[1]))
  print("{:-^75}".format(""))
  print("{:^10}".format("Agents") + "|" \
        + "{:^10}".format("Min, Max") + "|" \
        + "{:^50}".format("Object (Utility) [Max]") + "|")
  print("{:-^75}".format(""))
  for a in sorted(agent_prefs.keys()):
    print("{:^10}".format(str(a)) + "|" \
        + "{:^10}".format(str(min_object_per_reviewer) + "," + str(max_object_per_reviewer) + "| " \
        + ", ".join([str(o) + " (" + str(u) + ") " + "[" +str(agent_caps[a][o]) + "]" for o,u in sorted(agent_prefs[a].items(),key=lambda x:x[1], reverse=True)])))
  print("{:-^75}".format(""))

def pretty_print_utility_solution(m, agents, objects, assigned, utility, owa_utility=[]):
  print("{:-^75}".format(""))
  if m != 0:
    for a in agents:
      out = "Agent " + str(a) + " assigned : " + \
            ",".join([str(i) for i in objects if assigned[a,i].x >= 0.1]) + \
            " = " + str(utility[a].x)
      if owa_utility != []:
        out += " OWA: " + str(owa_utility[a].x)
      print(out)
    print("Finished in (seconds): " + str(m.Runtime))
    print("Objective Value: " + str(m.ObjVal))
  else:
    print("No Solution")
  print("{:-^75}".format(""))

def pretty_print_fractional_solution(m, agents, objects, assigned, utility, owa_utility=[]):
  print("{:-^75}".format(""))
  if m != 0:
    for a in agents:
      out = "Agent " + str(a) + " assigned : " + \
            ",".join([str(assigned[a,i].x) + " x " + str(i) for i in objects if assigned[a,i].x >= 0.001]) + \
            " = " + str(utility[a].x)
      if owa_utility != []:
        out += " OWA: " + str(owa_utility[a].x)
      print(out)
    print("Finished in (seconds): " + str(m.Runtime))
    print("Objective Value: " + str(m.ObjVal))
  else:
    print("No Solution")
  print("{:-^75}".format(""))

def pretty_print_rank_solution(m, agents, objects, assigned, rank_signature):
  print("{:-^75}".format(""))
  if m != 0:
    # Get ranks
    ranks = set([i for a,i in rank_signature.keys()])
    for a in agents:
      out = "Agent " + str(a) + " assigned : " + \
            ",".join([str(i) for i in objects if assigned[a,i].x >= 0.1]) + \
            " = " + str([str(rank_signature[a,r].x) for r in ranks])
      print(out)
    print("Finished in (seconds): " + str(m.Runtime))
    print("Objective Value: " + str(m.ObjVal))
  else:
    print("No Solution")
  print("{:-^75}".format(""))

def get_allocation_pickle(algorithm, agent_caps, agents, object_caps, objects, agent_prefs, assigned):
  '''
  Construct an object for pickletime...

  {
    algorithm: string with algorithm name for assignment
    agent_capacities: tuple (lower, upper)
    object_capacities: tuple (lower, upper)

    agents: list of agent names
    objects: list of items in allocation
    utilities: nested dicts: agent -> object -> value

    allocation: dict, agent -> list of items
  }

  Data
  -----------


  Notes
  ------------

  '''
  result = {}
  result["algorithm"] = algorithm
  result["agent_capacities"] = agent_caps
  result["object_capacities"] = object_caps

  result["agents"] = agents
  result["objects"] = objects

  result["allocation"] = {a:[] for a in agents}
  result["utilities"] = {a:{} for a in agents}

  for a in agents:
    result["allocation"][a] = [i for i in objects if assigned[a,i].x >= 0.1]
    result["utilities"][a] = {o: agent_prefs[a][o] for o in objects}

  return result

def read_preflib_file(fname):
  with open(fname, 'r') as input_file:
    l = input_file.readline()
    numcands = int(l.strip())
    candmap = {}
    for i in range(numcands):
      bits = input_file.readline().strip().split(",")
      candmap[int(bits[0].strip())] = bits[1].strip()

    #now we have numvoters, sumofvotecount, numunique orders
    bits = input_file.readline().strip().split(",")
    numvoters = int(bits[0].strip())
    sumvotes = int(bits[1].strip())
    uniqueorders = int(bits[2].strip())

    rankmaps = []
    rankmapcounts = []
    for i in range(uniqueorders):
      rec = input_file.readline().strip()
      #need to parse the rec properly..
      if rec.find("{") == -1:
        #its strict, just split on ,
        count = int(rec[:rec.index(",")])
        bits = rec[rec.index(",")+1:].strip().split(",")
        cvote = {}
        for crank in range(len(bits)):
          cvote[int(bits[crank])] = crank+1
        rankmaps.append(cvote)
        rankmapcounts.append(count)
      else:
        count = int(rec[:rec.index(",")])
        bits = rec[rec.index(",")+1:].strip().split(",")
        cvote = {}
        crank = 1
        partial = False
        for ccand in bits:
          if ccand.find("{") != -1:
            partial = True
            t = ccand.replace("{","")
            # make sure it isn't empty...
            if len(t) > 0 and t != '}':
              # make sure it isn't size one...
              if t.find("}") > -1:
                t = t.replace("}","")
              cvote[int(t.strip())] = crank
          elif ccand.find("}") != -1:
            partial = False
            t = ccand.replace("}","")
            if len(t) > 0:
              cvote[int(t.strip())] = crank
            crank += 1
          else:
            cvote[int(ccand.strip())] = crank
            if partial == False:
              crank += 1
        rankmaps.append(cvote)
        rankmapcounts.append(count)
  #Sanity check:
  if sum(rankmapcounts) != sumvotes or len(rankmaps) != uniqueorders:
    print("Error Parsing File: Votes Not Accounted For!")
    exit()

  # Convert this into the format we need...
  candmap = {x:y.replace(" ","") for x,y in candmap.items()}
  objects = list(candmap.values())
  agents = ["a"+str(i) for i in range(sum(rankmapcounts))]
  agent_prefs = {}

  # Impose Borda for now -- unrated get 0.
  # extract the number of ranks over all..
  #n_ranks = max([max(a.values()) for a in rankmaps])
  #scores = [5, 3, 1, 0]
  scores = list(range(len(candmap.keys())))
  scores.reverse()
  ctr = 0
  for rmap, count in zip(rankmaps, rankmapcounts):
    pref = {}
    for c_key,rank in rmap.items():
      pref[candmap[c_key]] = scores[rank-1]
    # items not appearing have value 0...
    rated = set([candmap[c] for c in rmap.keys()])
    allc = set(objects)
    notrated = allc - rated
    for n in notrated:
      pref[n] = 0
    for i in range(count):
      #copy it over...
      agent_prefs["a" + str(ctr)] = copy.copy(pref)
      ctr+=1

  return agents, objects, agent_prefs

'''
  Reader for the new format...
'''

def read_cost_file(fname):
  with open(fname, 'r') as input_file:
    li = input_file.readline().strip()
    while li.startswith("#"):
      #print(li)
      li = input_file.readline().strip()

    num_agents = int(li)
    num_objects = int(input_file.readline().strip())
    min_reviews_per_object = int(input_file.readline().strip())
    max_reviews_per_object = int(input_file.readline().strip())
    min_object_per_reviewer = int(input_file.readline().strip())
    max_object_per_reviewer = int(input_file.readline().strip())

    # Generate a bid structure.
    agents = ["a"+str(i) for i in range(num_agents)]
    objects = ["i"+str(i) for i in range(num_objects)]

    # Agent Prefs is a map from object --> value for each agent
    agent_prefs = {}
    for i,a in enumerate(agents):
      apref = {}
      bits = input_file.readline().strip().split(",")
      for j,o in enumerate(objects):
        apref[o] = float(bits[j])
      agent_prefs[a] = apref

    # Generate object cap vector
    object_caps = {}
    for i,o in enumerate(objects):
      object_caps[o] = (min_reviews_per_object, max_reviews_per_object)

    # Generate agent max
    agent_caps = {}
    for i,a in enumerate(agents):
      acap = {}
      bits = input_file.readline().strip().split(",")
      for j,o in enumerate(objects):
        acap[o] = float(bits[j])
      agent_caps[a] = acap

    return agents, objects, agent_prefs, agent_caps, object_caps, min_object_per_reviewer, max_object_per_reviewer











