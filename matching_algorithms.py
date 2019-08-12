import numpy as np
import copy
import os
import pathlib
import pickle
import sys
import gurobipy as gpy
sys.path.append('.\\allocation')
import build_models as sum_owa


class MatchingAlgorithm:
    def __init__(self, params):
        pass

    # assign the papers according to the algorithm and the bidding profile.
    def match(self, bidding_profile, params):
        print('Method not implemented')


class FractionalAllocation(MatchingAlgorithm):
    # A detailed explanation of this algorithm is available in the review_market.new2 pdf, pages 5-7.
    def match(self, bidding_profile, params):
        total_reviewers = params['total_reviewers']
        total_papers = params['total_papers']
        quota_matrix = params['quota_matrix']
        # step I
        prices = []
        for paper_index in range(0, total_papers):
            paper_demand = np.sum(bidding_profile, axis=0)[paper_index]
            paper_demand_count = np.sum(bidding_profile > 0, axis=0)[paper_index]
            # note change: if there are few bidders, even if they have high demand then the price is 1 since they should all get 1 unit of the paper
            if paper_demand_count <= params['papers_requirements'][paper_index]:
                paper_price = 1
            else:
                paper_price = min(1, ( params['papers_requirements'][paper_index] / paper_demand))
            prices.append(paper_price)
        fractional_allocation_profile = np.zeros((total_reviewers, total_papers))
        for reviewer_index in range(0, total_reviewers):
            for paper_index in range(0, total_papers):
                fractional_allocation_profile[reviewer_index][paper_index] = min(quota_matrix[reviewer_index][paper_index],
                                                (bidding_profile[reviewer_index][paper_index] * prices[paper_index]))
        first_step_allocation = copy.deepcopy(fractional_allocation_profile)
        # step II
        overbidders = []
        underbids = []
        k = sum(params['papers_requirements'])
        k = k / total_reviewers  # np.ceil(k / total_reviewers)
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
        papers_total_underbids = []
        for paper_index in range(0, total_papers):
            paper_total_underbid = (params['papers_requirements'][paper_index]
                                    - np.sum(fractional_allocation_profile, axis=0)[paper_index])
            papers_total_underbids.append(paper_total_underbid)
        U = sum(underbids)
        if U > 0:
            for reviewer_index in range(0, total_reviewers):
                for paper_index in range(0, total_papers):
                    if quota_matrix[reviewer_index][paper_index] == 0:
                        continue
                    else:
                        ## Reshef: I removed this part. All quota constraints are treated the same, whether the quota is 0 or 1 or something else
                        ##         the quota is ignored when deciding how much to allocate in step III, but any amount over the quota remains unallocated.
                        # underbids_with_coi = copy.deepcopy(underbids)
                        # for reviewer in range(0, total_reviewers):
                        #     if params['quota_matrix'][reviewer][paper_index] == 0:
                        #         underbids_with_coi[reviewer] = 0
                        # if sum(underbids_with_coi) == 0:
                        #     underbids_with_coi[reviewer_index] = 1
                        # fractional_allocation_profile[reviewer_index][paper_index] = \
                        #     min(params['quota_matrix'][reviewer_index][paper_index],
                        #         fractional_allocation_profile[reviewer_index][paper_index] +
                        #         papers_total_underbids[paper_index] * (underbids[reviewer_index] / sum(underbids_with_coi)))

                        fractional_allocation_profile[reviewer_index][paper_index] = \
                            min(quota_matrix[reviewer_index][paper_index],
                                fractional_allocation_profile[reviewer_index][paper_index] +
                                papers_total_underbids[paper_index] * (underbids[reviewer_index] / U))
        third_step_allocation = copy.deepcopy(fractional_allocation_profile)
        unallocated_papers = np.zeros(total_papers)
        for paper_index in range(0, total_papers):
            unallocated_papers[paper_index] = (params['papers_requirements'][paper_index]
                                               - np.sum(fractional_allocation_profile, axis=0)[paper_index])
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}


class FractionalSumOWA(MatchingAlgorithm):
    def match(self, bidding_profile, params):
        k = sum(params['papers_requirements'])
        k = k / params['total_reviewers']
        min_papers_per_reviewer = np.floor(k)
        max_papers_per_reviewer = np.ceil(k)

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
        #os.remove('gurobi.log')
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}


class DiscreteSumOWA(MatchingAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.type = None
        self.file_extension = None

    def match(self, bidding_profile, params):
        common_and_unique_bids = self.adjust_input_format(bidding_profile, params)
        k = sum(params['papers_requirements'])
        k = k / params['total_reviewers']  # np.ceil(k / total_reviewers) ??????????
        minimum_papers_per_reviewer = int(np.floor(k))
        maximum_papers_per_reviewer = int(np.ceil(k))
        minimum_reviewers_per_paper = int(np.floor(min(params['papers_requirements'])))
        maximum_reviewers_per_paper = int(np.floor(max(params['papers_requirements'])))
        os.system('python .\\allocation\\cap_discrete_alloc.py -d .\\output\\tmp_input_adjust999.toi -p '
                  '.\\output\\tmp_output_adjust999 -a ' + str(minimum_papers_per_reviewer) + ' -A ' +
                  str(maximum_papers_per_reviewer) + ' -o ' + str(minimum_reviewers_per_paper) + ' -O ' +
                  str(maximum_reviewers_per_paper) + ' ' + self.type)
        algorithm_output = self.adjust_output_format(common_and_unique_bids, params)
        os.remove('.\\output\\tmp_input_adjust999.toi')
        os.remove('.\\output\\tmp_output_adjust999' + self.file_extension + '.pickle')
        #os.remove('gurobi.log')
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

    def adjust_input_format(self, bidding_profile, params):
        try:
            pathlib.Path('.\\output').mkdir()
        except FileExistsError:
            pass
        path = '.\\output\\tmp_input_adjust999.toi'
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

    def adjust_output_format(self, common_and_unique_bids, params):
        first_step_allocation = np.zeros((params['total_reviewers'], params['total_papers']))
        for reviewer in range(0, params['total_reviewers']):
            for paper in range(0, params['total_papers']):
                first_step_allocation[reviewer][paper] = -1
        second_step_allocation = first_step_allocation
        owa_algo_agents_to_reviewers_map = []
        for common_bid in common_and_unique_bids[0]:
            owa_algo_agents_to_reviewers_map = owa_algo_agents_to_reviewers_map + common_bid[1]
        owa_algo_agents_to_reviewers_map = owa_algo_agents_to_reviewers_map + common_and_unique_bids[1]
        with open('.\\output\\tmp_output_adjust999' + self.file_extension + '.pickle', "rb") as openfile:
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
                       'FractionalSumOWA': FractionalSumOWA,
                       'Utilitarian': Utilitarian,
                       'Egalitarian': Egalitarian,
                       'RankMaximal': RankMaximal,
                       'LinearSumOWA': LinearSumOWA,
                       'Nash': Nash}





