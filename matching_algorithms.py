import numpy as np
import copy
import os
import pathlib
import pickle


class MatchingAlgorithm:
    def __init__(self, params):
        pass

    # assign the papers according to the algorithm and the bidding profile.
    def match(self, bidding_profile, params):
        print('Method not implemented')

    # adjust the given bidding profile into the required format for the selected algorithm.
    def adjust_input_format(self, bidding_profile, params):
        print('Method not implemented')

    # adjust the output of the selected algorithm to be the same as the FractionAllocation algorithm output.
    def adjust_output_format(self, additional_info, params):
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
            if paper_demand == 0:
                paper_price = 1
            else:
                paper_price = min(1, (params['papers_requirements'][paper_index] / paper_demand))
            prices.append(paper_price)
        fractional_allocation_profile = np.zeros((total_reviewers, total_papers))
        for reviewer_index in range(0, total_reviewers):
            for paper_index in range(0, total_papers):
                fractional_allocation_profile[reviewer_index][paper_index] = min(quota_matrix[reviewer_index][paper_index],
                                                (bidding_profile[reviewer_index][paper_index] *  prices[paper_index]))
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
        if U>0:
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

    def adjust_input_format(self, bidding_profile, params):
        pass

    def adjust_output_format(self, algorithm_output, params):
        pass


class SumOWA(MatchingAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.type = params['matching_algorithm'][-2:]
        if self.type == '-u':
            self.file_extension = '-utilitarian'
        elif self.type == '-e':
            self.file_extension = '-egalitarian'
        elif self.type == '-r':
            self.file_extension = '-rank_maximal'
        elif self.type == '-l':
            self.file_extension = '-linearsumowa'
        elif self.type == '-n':
            self.file_extension = '-nash'
        else:
            self.file_extension = None

    def match(self, bidding_profile, params):
        common_and_unique_bids = self.adjust_input_format(bidding_profile, params)
        k = sum(params['papers_requirements'])
        k = k / params['total_reviewers']  # np.ceil(k / total_reviewers) ??????????
        k = int(k)
        os.system('python .\\allocation\\cap_discrete_alloc.py -d .\\output\\tmp_input_adjust999.toi -p '
                  '.\\output\\tmp_output_adjust999 -a ' + str(k) + ' -A ' + str(k) + ' -o ' +
                  str(min(params['papers_requirements'])) + ' -O ' + str(max(params['papers_requirements'])) + ' '
                  + self.type)
        algorithm_output = self.adjust_output_format(common_and_unique_bids, params)
        os.remove('.\\output\\tmp_input_adjust999.toi')
        os.remove('.\\output\\tmp_output_adjust999' + self.file_extension + '.pickle')
        os.remove('gurobi.log')
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
        bidded = [paper for paper, amount in enumerate(bid) if amount == 1]
        didnt_bid = [paper for paper, amount in enumerate(bid) if amount == 0]
        for paper in didnt_bid:
            if params['quota_matrix'][reviewers[0]][paper] == 0:
                didnt_bid.remove(paper)
        bidded = str(bidded)[1:-1]
        didnt_bid = str(didnt_bid)[1:-1]
        file.write('{0},{{1}},{{2}}\n'.format(len(reviewers), bidded, didnt_bid))

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
                third_step_allocation[reviewer][int(paper[-1:])] = 1
        unallocated_papers = np.zeros(params['total_papers'])
        for paper_index in range(0, params['total_papers']):
            unallocated_papers[paper_index] = (params['papers_requirements'][paper_index]
                                               - np.sum(third_step_allocation, axis=0)[paper_index])
        return {'first_step_allocation': first_step_allocation, 'second_step_allocation': second_step_allocation,
                'third_step_allocation': third_step_allocation, 'unallocated_papers': unallocated_papers}


possible_algorithms = {'FractionalAllocation': FractionalAllocation, 'SumOWA-u': SumOWA, 'SumOWA-e': SumOWA,
                       'SumOWA-r': SumOWA, 'SumOWA-l': SumOWA, 'SumOWA-n': SumOWA}





# example from pdf:
# bid_p = [[1, 0, 1, 0, 0, 0],
#          [1, 1, 1, 1, 0, 0],
#          [0, 1, 1, 0, 1, 0],
#          [0, 1, 1, 1, 0, 0],
#          [1, 1, 1, 0, 0, 0]]
#
# p = Instance(None, 6, 5, 2)
# res = fractional_allocation_algorithm(bid_p, p)
# print(res[2])
#
# import itertools
# from instance_generator import *
# algo = FractionalAllocation()
# bid_p = [[1, 1, 1, 0, 0, 0],
#          [0, 0, 0, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0]]
# private_costs = [0, 0, 1, 1, 2, 2]
# best_responses = []
# coi = [[], [], [0, 1]]
# tmp_instance = Instance(coi, None, 6, 3, 1, None, None)
# possible_papers = [x for x in range(0, 6) if x not in coi[2]]
# interesting_bids = [(2,)]
# interesting_results = []
# for num_of_bids in range(0, 7):
#     for paper_bids in itertools.combinations(possible_papers, num_of_bids):
#         for i in range(0, 6):
#             if i in paper_bids:
#                 bid_p[2][i] = 1
#             else:
#                 bid_p[2][i] = 0
#         allocation = algo.match(bid_p, tmp_instance)['third_step_allocation']
#         cost = allocation[2][2] + allocation[2][3] + 2 * allocation[2][4] + 2 * allocation[2][5]
#         if len(best_responses) == 0 or best_responses[0][0] == cost:
#             best_responses.append((cost, paper_bids, allocation))
#         elif best_responses[0][0] > cost:
#             best_responses = [(cost, paper_bids, allocation)]
#         if paper_bids in interesting_bids:
#             interesting_results.append((cost, paper_bids, allocation))
#
# print('best cost:{0}'.format(best_responses[0][0]))
# print('best bid:{0}'.format(best_responses[0][1]))
# print('best allocation:{0}'.format(best_responses[0][2]))
#
# for bid in interesting_results:
#     print('cost:{0}'.format(bid[0]))
#     print('bid:{0}'.format(bid[1]))
#     print('allocation:{0}'.format(bid[2]))

# from instance_generator import *
# algo = FractionalAllocation()
# bid_p = [[1, 0, 0],
#          [0, 1, 1],
#          [0, 1, 1]]
# coi = [[], [], []]
# tmp_instance = Instance(coi, None, 3, 3, 1, None, None)
# allocation = algo.match(bid_p, tmp_instance)
# print(allocation['first_step_allocation'])
# print(allocation['second_step_allocation'])
# print(allocation['third_step_allocation'])
# print(allocation['unallocated_papers'])


