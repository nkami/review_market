import numpy as np
import copy
import os
import pathlib
import pickle
import sys
import gurobipy as gpy
import warnings
import datetime
import time
sys.path.append(os.path.join(".", "allocation"))
#import build_models as sum_owa


class MatchingAlgorithm:
    def __init__(self, params):
        pass

    # assign the papers according to the algorithm and the bidding profile.
    def match(self, bidding_profile, params):
        print('Method not implemented')

## A fractional algorithm that returns a valid and complete assignment. Does not coincide with the one in the paper.
class FractionalAllocation(MatchingAlgorithm):
    def match(self, bidding_profile, params):
        total_reviewers = params['total_reviewers']
        total_papers = params['total_papers']
        quota_matrix = params['quota_matrix']
        paper_req = params['papers_requirements']
        # step I
        prices = []
        for paper_index in range(0, total_papers):
            paper_demand = np.sum(bidding_profile, axis=0)[paper_index]
            paper_demand_count = np.sum(bidding_profile > 0, axis=0)[paper_index]
            # note change: if there are few bidders, even if they have high demand then the price is 1 since they should all get 1 unit of the paper
            if paper_demand_count <= paper_req[paper_index]:
                paper_price = 1
            else:
                paper_price = min(1, (paper_req[paper_index] / paper_demand))
            prices.append(paper_price)
        fractional_allocation_profile = np.zeros((total_reviewers, total_papers))
        for reviewer_index in range(0, total_reviewers):
            for paper_index in range(0, total_papers):
                fractional_allocation_profile[reviewer_index][paper_index] = min(
                    quota_matrix[reviewer_index][paper_index],
                    (bidding_profile[reviewer_index][paper_index] * prices[paper_index]))
        first_step_allocation = copy.deepcopy(fractional_allocation_profile)
        # step II
        overbidders = []
        underbids = []
        k = sum(paper_req) / total_reviewers  # np.ceil(k / total_reviewers)
        for reviewer_index in range(0, total_reviewers):
            overbid_of_reviewer = np.sum(fractional_allocation_profile, axis=1)[reviewer_index] - k
            if overbid_of_reviewer > 0:
                overbidders.append((reviewer_index, overbid_of_reviewer))
                underbids.append(0)
            else:
                underbids.append(abs(overbid_of_reviewer))
        for overbidder in overbidders:
            for paper_index in range(0, total_papers):
                fractional_allocation_profile[overbidder[0]][paper_index] *= (k / (k + overbidder[1]))
        second_step_allocation = copy.deepcopy(fractional_allocation_profile)
        # step III
        paper_total_underbids = []
        free_space_matrix = np.zeros((total_reviewers, total_papers))
        for paper_index in range(0, total_papers):
            paper_total_underbid = (paper_req[paper_index]
                                    - np.sum(fractional_allocation_profile, axis=0)[paper_index])
            paper_total_underbids.append(paper_total_underbid)
            for reviewer_index in range(0, total_reviewers):
                free_space_matrix[reviewer_index][paper_index] = \
                    np.min([underbids[reviewer_index],
                           quota_matrix[reviewer_index][paper_index]-fractional_allocation_profile[reviewer_index][paper_index]])
            # if free_space_matrix[:,paper_index].sum() < paper_total_underbid:
            #     print(paper_index)
            #     print(free_space_matrix[:,paper_index].sum() )
            #     print(paper_total_underbid)
        if sum(underbids) > 0:
            free_space_for_paper = np.sum(free_space_matrix, axis=0)
            for paper_index in range(0, total_papers):
                current_bids = np.sum(fractional_allocation_profile, axis=1)
                free_space_for_this_paper = free_space_for_paper[paper_index]
                current_free_space_for_paper = []
                for reviewer_index in range(0, total_reviewers):
                    current_free_space_for_paper.append(np.min([k-current_bids[reviewer_index],
                                                                quota_matrix[reviewer_index][paper_index]-fractional_allocation_profile[reviewer_index][paper_index]]))
                sum_free_space_for_paper = np.sum(current_free_space_for_paper)
                if sum_free_space_for_paper < paper_total_underbid-0.0001:
                    print(sum_free_space_for_paper)
                    print(paper_total_underbid)
                    print(paper_index)

                for reviewer_index in range(0, total_reviewers):
                    if quota_matrix[reviewer_index][paper_index] == 0:
                        continue
                    else:
                        fractional_allocation_profile[reviewer_index][paper_index] = fractional_allocation_profile[reviewer_index][paper_index] + paper_total_underbids[paper_index] * current_free_space_for_paper[reviewer_index]/sum_free_space_for_paper
             #      TRY2:         fractional_allocation_profile[reviewer_index][paper_index] +   paper_total_underbids[paper_index] * (free_space_matrix[reviewer_index][paper_index] / free_space_for_this_paper)


                        # TRY1: min(quota_matrix[reviewer_index][paper_index],
                        #    fractional_allocation_profile[reviewer_index][paper_index] +
                        #    papers_total_underbids[paper_index] * (underbids[reviewer_index] / sum(underbids)))
        third_step_allocation = copy.deepcopy(fractional_allocation_profile)
        unallocated_papers = np.zeros(total_papers)
        for paper_index in range(0, total_papers):
            unallocated_papers[paper_index] = (paper_req[paper_index]
                                               - np.sum(fractional_allocation_profile, axis=0)[paper_index])
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}


class FractionalSumOWA(MatchingAlgorithm):
    def match(self, bidding_profile, params):
        # k = sum(params['papers_requirements'])
        # k = k / params['total_reviewers']
        # min_papers_per_reviewer = np.floor(k)
        # max_papers_per_reviewer = np.ceil(k)

        min_papers_per_reviewer = 3
        max_papers_per_reviewer = 4

        # Generate a bid structure.
        agents = ["a" + str(i) for i in range(params['total_reviewers'])]
        papers = ["i" + str(i) for i in range(params['total_papers'])]

        # Agent Prefs is a map from object --> value for each agent
        agent_prefs = {}
        for i, a in enumerate(agents):
            apref = {}
            bits = [str(bid) for bid in bidding_profile[i]]
            for j, o in enumerate(papers):
                apref[o] = float(bits[j])
            agent_prefs[a] = apref

        # Generate object cap vector
        papers_caps = {}
        for i, o in enumerate(papers):
            min_reviewers_per_paper = np.floor(params['papers_requirements'][i])
            max_reviewers_per_paper = np.ceil(params['papers_requirements'][i])
            papers_caps[o] = (min_reviewers_per_paper, max_reviewers_per_paper)

        # Generate agent max
        agent_caps = {}
        for i, a in enumerate(agents):
            acap = {}
            bits = [str(quota) for quota in params['quota_matrix'][i]]
            for j, o in enumerate(papers):
                acap[o] = float(bits[j])
            agent_caps[a] = acap

        model, assigned, utility = sum_owa.max_fractional_model(agents, papers, agent_prefs, agent_caps, papers_caps,
                                                                min_papers_per_reviewer, max_papers_per_reviewer)
        model.setParam(gpy.GRB.Param.OutputFlag, False)
        model.optimize()

        first_step_allocation = np.zeros((params['total_reviewers'], params['total_papers']))
        third_step_allocation = np.zeros((params['total_reviewers'], params['total_papers']))
        for reviewer in range(0, params['total_reviewers']):
            for paper in range(0, params['total_papers']):
                first_step_allocation[reviewer][paper] = -1
                third_step_allocation[reviewer][paper] = assigned['a' + str(reviewer), 'i' + str(paper)].x
        second_step_allocation = first_step_allocation
        unallocated_papers = np.zeros(params['total_papers'])
        for paper in range(0, params['total_papers']):
            unallocated_papers[paper] = (params['papers_requirements'][paper]
                                         - np.sum(third_step_allocation, axis=0)[paper])
        # os.remove('gurobi.log')
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}


## An implementation of the algorithm in the paper.
## The mock algorithm does not return a valid assignment, since each agent is allocated independently and thus more than r copies of a paper may be allocated.
class MockAllocation(MatchingAlgorithm):
    def match(self, bidding_profile, params):
        total_reviewers = params['total_reviewers']
        total_papers = params['total_papers']
        quota_matrix = params['quota_matrix']
        paper_req = params['papers_requirements']
        # step I: same as in fractional algorithm
        prices = []
        for paper_index in range(0, total_papers):
            paper_demand = np.sum(bidding_profile, axis=0)[paper_index]
            paper_demand_count = np.sum(bidding_profile > 0, axis=0)[paper_index]
            if paper_demand_count <= paper_req[paper_index]:
                paper_price = 1
            else:
                paper_price = min(1, (paper_req[paper_index] / paper_demand))
            prices.append(paper_price)
        fractional_allocation_profile = np.zeros((total_reviewers, total_papers))
        for reviewer_index in range(0, total_reviewers):
            for paper_index in range(0, total_papers):
                fractional_allocation_profile[reviewer_index][paper_index] = min(
                    quota_matrix[reviewer_index][paper_index],
                    (bidding_profile[reviewer_index][paper_index] * prices[paper_index]))
        first_step_allocation = copy.deepcopy(fractional_allocation_profile)
        # step II O: same as in fractional algorithm
        overbidders = []
        overbids = []   #  of all n bidders
        underbids = []  # of all n bidders
        k = sum(paper_req) / total_reviewers
        for reviewer_index in range(0, total_reviewers):
            overbid_of_reviewer = np.sum(fractional_allocation_profile, axis=1)[reviewer_index] - k
            if overbid_of_reviewer > 0:
                overbidders.append(reviewer_index)
                overbids.append(overbid_of_reviewer)
                underbids.append(0)
            else:
                overbids.append(0)
                underbids.append(abs(overbid_of_reviewer))
        for reviewer_index in range(0,total_reviewers ):
            if reviewer_index in overbidders:
                for paper_index in range(0, total_papers):
                    fractional_allocation_profile[reviewer_index][paper_index] *= (k / (k + overbids[reviewer_index]))
        second_step_allocation = copy.deepcopy(fractional_allocation_profile)
        # step II U
        # compute allocation for each bidder independently:
        for reviewer_index in range(0, total_reviewers):
            if reviewer_index in overbidders:
                continue
            paper_excess = np.zeros(total_papers)  # u_j
            constrained_papers = []    # Q
            for paper_index in range(0, total_papers):
                paper_excess[paper_index] = paper_req[paper_index] - np.sum(first_step_allocation, axis=0)[paper_index]
                if first_step_allocation[reviewer_index][paper_index] >= quota_matrix[reviewer_index][paper_index]:
                    constrained_papers.append(paper_index)
            while 1:
                new_constrained_papers = []   # Q+
                current_allocated = np.zeros(total_papers)  #  y_j
                current_excess_papers = 0
                for paper_index in range(0, total_papers):
                    if paper_index in constrained_papers:
                        current_allocated[paper_index] = quota_matrix[reviewer_index][paper_index]
                    else:
                        current_allocated[paper_index] = first_step_allocation[reviewer_index][paper_index]
                        current_excess_papers += paper_excess[paper_index]
                current_free_space = k-np.sum(current_allocated)
                total_to_allocate = np.max([current_excess_papers,current_free_space])  #  su
                for paper_index in range(0, total_papers):
                    if paper_index in constrained_papers:
                        fractional_allocation_profile[reviewer_index][paper_index] = quota_matrix[reviewer_index][paper_index]
                    else:
                        fractional_allocation_profile[reviewer_index][paper_index] = current_allocated[paper_index] + paper_excess[paper_index] * current_free_space / total_to_allocate
                        if fractional_allocation_profile[reviewer_index][paper_index] > quota_matrix[reviewer_index][paper_index]:
                            new_constrained_papers.append(paper_index)
                if not new_constrained_papers:
                    break
                constrained_papers.extend(new_constrained_papers)
        third_step_allocation = copy.deepcopy(fractional_allocation_profile)
        unallocated_papers = np.zeros(total_papers)
        for paper_index in range(0, total_papers):
            unallocated_papers[paper_index] = (paper_req[paper_index]
                                               - np.sum(fractional_allocation_profile, axis=0)[paper_index])
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}


## A valid version of the Mock Algorithm, that does not allocate all papers.
class MockAllocationAll(MatchingAlgorithm):
    def match(self, bidding_profile, params):
        total_reviewers = params['total_reviewers']
        total_papers = params['total_papers']
        quota_matrix = params['quota_matrix']
        paper_req = params['papers_requirements']
        # step IA: compute an Initial Assignment
        prices = []
        for paper_index in range(0, total_papers):
            paper_demand = np.sum(bidding_profile, axis=0)[paper_index]
            paper_demand_count = np.sum(bidding_profile > 0, axis=0)[paper_index]
            if paper_demand_count <= paper_req[paper_index]:
                paper_price = 1
            else:
                paper_price = min(1, (paper_req[paper_index] / paper_demand))
            prices.append(paper_price)
        fractional_allocation_profile = np.zeros((total_reviewers, total_papers))
        for reviewer_index in range(0, total_reviewers):
            for paper_index in range(0, total_papers):
                fractional_allocation_profile[reviewer_index][paper_index] = min(
                    quota_matrix[reviewer_index][paper_index],
                    (bidding_profile[reviewer_index][paper_index] * prices[paper_index]))
        first_step_allocation = copy.deepcopy(fractional_allocation_profile)
        # step OB: same as in fractional algorithm
        overbidders = []
        overbids = []   #  of all n bidders
        underbids = []  # of all n bidders
        k = sum(paper_req) / total_reviewers
        for reviewer_index in range(0, total_reviewers):
            overbid_of_reviewer = np.sum(fractional_allocation_profile, axis=1)[reviewer_index] - k
            if overbid_of_reviewer > 0:
                overbidders.append(reviewer_index)
                overbids.append(overbid_of_reviewer)
                underbids.append(0)
            else:
                overbids.append(0)
                underbids.append(abs(overbid_of_reviewer))
        for reviewer_index in range(0,total_reviewers ):
            if reviewer_index in overbidders:
                for paper_index in range(0, total_papers):
                    fractional_allocation_profile[reviewer_index][paper_index] *= (k / (k + overbids[reviewer_index]))
        second_step_allocation = copy.deepcopy(fractional_allocation_profile)
        # step UB
        # compute allocation for each bidder independently:
        constrained_papers = [] # tuples of (i,j)
        
        for reviewer_index in range(0, total_reviewers):
            for paper_index in range(0, total_papers):
                if first_step_allocation[reviewer_index][paper_index] >= quota_matrix[reviewer_index][paper_index]:
                    constrained_papers.append((reviewer_index, paper_index))
        while 1:
            fractional_allocation_profile = copy.deepcopy(second_step_allocation) # revert allocation
            # assign all constrained papers:
            for (reviewer_index,paper_index) in constrained_papers:
                if reviewer_index not in overbidders:
                   fractional_allocation_profile[reviewer_index][paper_index] = quota_matrix[reviewer_index][paper_index]
            paper_allocated = np.sum(fractional_allocation_profile,axis=0)
            reviewer_allocated = np.sum(fractional_allocation_profile,axis=1)
            paper_excess = np.subtract(paper_req,paper_allocated)   # u_j after step II and constrained papers

            new_constrained_papers = []  # Q+
            for reviewer_index in range(0, total_reviewers):
                if reviewer_index in overbidders:
                    continue
                current_free_space = k-reviewer_allocated[reviewer_index]
                unconstrained_excess_papers = 0
                for paper_index in range(0,total_papers):
                    if (reviewer_index,paper_index) not in constrained_papers:
                        unconstrained_excess_papers += paper_excess[paper_index]
                total_to_allocate = np.max([unconstrained_excess_papers,current_free_space])  #  su
                # if unconstrained_excess_papers < current_free_space-0.00001:
                #     print(current_free_space-unconstrained_excess_papers)
                added = np.zeros(total_papers)
                for paper_index in range(0, total_papers):
                    if (reviewer_index,paper_index) not in constrained_papers:
                        added[paper_index] = paper_excess[paper_index] * current_free_space / total_to_allocate
                        fractional_allocation_profile[reviewer_index][paper_index] = first_step_allocation[reviewer_index][paper_index] \
                                                                                     + added[paper_index]
                        if fractional_allocation_profile[reviewer_index][paper_index] >= quota_matrix[reviewer_index][paper_index]:
                            new_constrained_papers.append((reviewer_index,paper_index))
                # if np.sum(fractional_allocation_profile[reviewer_index]) < k-0.00001:
                #     print(k-np.sum(fractional_allocation_profile[reviewer_index]))
            if not new_constrained_papers:
                break
            constrained_papers.extend(new_constrained_papers)
        third_step_allocation = copy.deepcopy(fractional_allocation_profile)
        unallocated_papers = np.zeros(total_papers)
        for paper_index in range(0, total_papers):
            unallocated_papers[paper_index] = (paper_req[paper_index]
                                               - np.sum(fractional_allocation_profile, axis=0)[paper_index])
        percent_unallocated = 100*np.sum(unallocated_papers)/np.sum(paper_req)
        if percent_unallocated > 0.001:
            warnings.warn("{0}% of papers unallocated\n".format(percent_unallocated))
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}

class DiscreteSumOWA(MatchingAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.type = None
        self.file_extension = None

    def match(self, bidding_profile, params):
        time_stamp = datetime.datetime.now().isoformat()[:-3].replace(':', '-')
        # NSM: MAKE THIS PLATFORM INDEPENDENT!
        #input_tmp_filename = ".\\output\\tmp_input_adjust{0}.toi".format(time_stamp)
        input_tmp_filename = os.path.join(".", "Output", "tmp_input_adjust{0}.toi".format(time_stamp))
        #output_tmp_filename = ".\\output\\tmp_output_adjust{0}".format(time_stamp)
        output_tmp_filename = os.path.join(".", "Output", "tmp_output_adjust{0}".format(time_stamp))
        
        common_bids, unique_bidders = self.adjust_input_format(bidding_profile, params,input_tmp_filename)
        k = sum(params['papers_requirements'])
        k = k / params['total_reviewers']
        minimum_papers_per_reviewer = int(np.floor(k))
        maximum_papers_per_reviewer = int(np.ceil(k))
        minimum_reviewers_per_paper = int(np.floor(min(params['papers_requirements'])))
        maximum_reviewers_per_paper = int(np.floor(max(params['papers_requirements'])))
        os.system('echo %GRB_LICENSE_FILE%')

        pythonpath = os.path.join(".", "allocation", "cap_discrete_alloc.py")
        os.system("python3 " + pythonpath + " -d {0} -p {1} -a ".format(input_tmp_filename,output_tmp_filename) + str(minimum_papers_per_reviewer) + ' -A ' +   str(maximum_papers_per_reviewer) + ' -o ' + str(minimum_reviewers_per_paper) + ' -O ' + str(maximum_reviewers_per_paper) + ' ' + self.type)

        algorithm_output = self.adjust_output_format(common_bids, unique_bidders, params, output_tmp_filename)
        os.remove(input_tmp_filename)
        os.remove(output_tmp_filename + self.file_extension + '.pickle')
        # os.remove('gurobi.log')
        return algorithm_output

    # the bids1-2 are tuples of: (bid, bidder id)
    def check_if_bids_equal(self, bid1, bid2, params):
        equal = True
        diff = [abs(bid1[0][paper] - bid2[0][paper]) for paper in range(0, params['total_papers'])]
        if sum(diff) != 0:
            equal = False
        for paper in range(0, params['total_papers']):  # check that COI is the same as well
            if params['quota_matrix'][bid1[1]][paper] != params['quota_matrix'][bid2[1]][paper]:
                equal = False
        return equal

    def write_bid_to_file(self, bid, reviewers, file, params):
        bidded_one = [paper for paper, amount in enumerate(bid) if amount == 1]
        bidded_two = [paper for paper, amount in enumerate(bid) if amount == 2]
        didnt_bid = [paper for paper, amount in enumerate(bid) if amount == 0]
        for paper in didnt_bid:
            if params['quota_matrix'][reviewers[0]][paper] == 0:
                didnt_bid.remove(paper)
        bidded_one = str(bidded_one)[1:-1]
        bidded_two = str(bidded_two)[1:-1]
        didnt_bid = str(didnt_bid)[1:-1]
        file.write(str(len(reviewers)) + ',{' + bidded_two + '},{' + bidded_one + '},{' + didnt_bid + '}\n')

    def adjust_input_format(self, bidding_profile, params,input_filename):
        try:
            pathlib.Path(os.path.join(".", "output")).mkdir()
        except FileExistsError:
            pass
        path = input_filename
        with open(path, 'w') as output_file:
            output_file.write('{0}\n'.format(params['total_papers']))
            for paper in range(0, params['total_papers']):
                output_file.write('{0},paper {1}\n'.format(paper, paper))
            common_bids = []  # a list of tuples: (bid, list of reviewers)
            unique_bidders = [reviewer for reviewer in range(0, params['total_reviewers'])]
            for reviewer_bid in range(0, params['total_reviewers']):
                for current_bid in range(0, params['total_reviewers']):
                    bid1 = (bidding_profile[reviewer_bid], reviewer_bid)
                    bid2 = (bidding_profile[current_bid], current_bid)
                    if reviewer_bid != current_bid and self.check_if_bids_equal(bid1, bid2, params):
                        in_common_bids = False
                        unique_bidders.remove(reviewer_bid)
                        for bid in common_bids:
                            if self.check_if_bids_equal(bid1, (bid[0], bid[1][0]), params):
                                bid[1].append(reviewer_bid)
                                in_common_bids = True
                                break
                        if not in_common_bids:
                            common_bids.append((bidding_profile[reviewer_bid], [reviewer_bid]))
                        break
            output_file.write('{0},{1},{2}\n'.format(params['total_reviewers'], params['total_reviewers'],
                                                     len(unique_bidders) + len(common_bids)))
            for bid in common_bids:
                self.write_bid_to_file(bid[0], bid[1], output_file, params)
            for bid in unique_bidders:
                self.write_bid_to_file(bidding_profile[bid], [bid], output_file, params)
        return common_bids, unique_bidders

    def adjust_output_format(self, common_bids, unique_bidders, params ,output_filename):
        first_step_allocation = np.zeros((params['total_reviewers'], params['total_papers']))
        for reviewer in range(0, params['total_reviewers']):
            for paper in range(0, params['total_papers']):
                first_step_allocation[reviewer][paper] = np.nan
        second_step_allocation = first_step_allocation
        owa_algo_agents_to_reviewers_map = []
        for common_bid in common_bids:
            owa_algo_agents_to_reviewers_map = owa_algo_agents_to_reviewers_map + common_bid[1]
        owa_algo_agents_to_reviewers_map = owa_algo_agents_to_reviewers_map + unique_bidders
        with open(output_filename + self.file_extension + '.pickle', "rb") as openfile:
            file_info = pickle.load(openfile)
        third_step_allocation = np.zeros((params['total_reviewers'], params['total_papers']))
        for agent, reviewer in enumerate(owa_algo_agents_to_reviewers_map):
            for paper in file_info['allocation']['a{0}'.format(agent)]:
                paper_idx = int(paper.replace('paper', ''))
                third_step_allocation[reviewer][paper_idx] = 1
        unallocated_papers = np.zeros(params['total_papers'])
        for paper_index in range(0, params['total_papers']):
            unallocated_papers[paper_index] = (params['papers_requirements'][paper_index]
                                               - np.sum(third_step_allocation, axis=0)[paper_index])
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}


class Utilitarian(DiscreteSumOWA):
    def __init__(self, params):
        super().__init__(params)
        self.type = '-u'
        self.file_extension = '-utilitarian'


class Egalitarian(DiscreteSumOWA):
    def __init__(self, params):
        super().__init__(params)
        self.type = '-e'
        self.file_extension = '-egalitarian'


class RankMaximal(DiscreteSumOWA):
    def __init__(self, params):
        super().__init__(params)
        self.type = '-r'
        self.file_extension = '-rank_maximal'


class LinearSumOWA(DiscreteSumOWA):
    def __init__(self, params):
        super().__init__(params)
        self.type = '-l'
        self.file_extension = '-linearsumowa'


class Nash(DiscreteSumOWA):
    def __init__(self, params):
        super().__init__(params)
        self.type = '-n'
        self.file_extension = '-nash'


possible_algorithms = {'FractionalAllocation': FractionalAllocation,
                        'MockAllocation': MockAllocation,
                       'MockAllocationAll': MockAllocationAll,
                       'FractionalSumOWA': FractionalSumOWA,
                       'Utilitarian': Utilitarian,
                       'Egalitarian': Egalitarian,
                       'RankMaximal': RankMaximal,
                       'LinearSumOWA': LinearSumOWA,
                       'Nash': Nash}
