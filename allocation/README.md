# Discrete Allocations
README.md (c) Nicholas Mattei - Tulane University

## OVERVIEW

This Repo contains all the code needed to run the experiments for our paper for Sum-OWAs.  If you use code or data from this repo please cite our paper.

```
@inproceedings{LiMaNoWa18,
    Author = {J. W. Lian and N. Mattei and R. Noble and T. Walsh},
    Booktitle = {Proc. of the 32nd AAAI Conference on Artificial Intelligence (AAAI)},
    Title = {The Conference Paper Assignment Problem: Using Order Weighted Averages to Assign Indivisible Goods},
    Year = {2018}}
```

## DETAILS

This script can be used to compute assignments and various metrics of an allocation and save the results to a particular file.

Example Interaction
```
/discrete_allocation.git/src$ /usr/bin/python3 cap_discrete_alloc.py -d ./toy_model.toc -a 2 -A 5 -o 2 -O 5
3
finished
Utilitarian
---------------------------------------------------------------------------
Agent a0 assigned : House1,House2,House3,House4,House5 = 17.0
Agent a1 assigned : House1,House3,House4,House5,House6 = 20.0
Agent a2 assigned : House2,House3,House4,House5,House6 = 16.0
Finished in (seconds): 0.00180506706238
Objective Value: 53.0
---------------------------------------------------------------------------


```

The file will dump a pickle file of the result which has the following format.

```
{
    algorithm: string with algorithm name for assignment
    agent_capacities: tuple (lower, upper)
    object_capacities: tuple (lower, upper)
    
    agents: list of agent names
    objects: list of items in allocation
    utilities: nested dicts: agent -> object -> value

    allocation: dict, agent -> list of items
}


## Requirements

This requires Python3 and Gurobi 7+ to run.  A free academic license and instructions on how to install are available here: (http://www.gurobi.com/)
