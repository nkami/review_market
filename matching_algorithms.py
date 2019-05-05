import numpy as np
import copy


class MatchingAlgorithm:
    # assign the papers according to the algorithm and the bidding profile.
    def match(self, bidding_profile, params):
        print('Method not implemented')

    # adjust the given bidding profile into the required format for the selected algorithm.
    def adjust_input_format(self, bidding_profile, params):
        print('Method not implemented')

    # adjust the output of the selected algorithm to be the same as the FractionAllocation algorithm output.
    def adjust_output_format(self, algorithm_output, params):
        print('Method not implemented')


class FractionalAllocation(MatchingAlgorithm):
    # A detailed explanation of this algorithm is available in the review_market.new2 pdf, pages 5-7.
    def match(self, bidding_profile, params):
        total_reviewers = params['total_reviewers']
        total_papers = params['total_papers']
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
                fractional_allocation_profile[reviewer_index][paper_index] = (
                            bidding_profile[reviewer_index][paper_index] *
                            prices[paper_index])
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
        for reviewer_index in range(0, total_reviewers):
            for paper_index in range(0, total_papers):
                if params['quota_matrix'][reviewer_index][paper_index] == 0:
                    continue
                else:
                    underbids_with_coi = copy.deepcopy(underbids)
                    for reviewer in range(0, total_reviewers):
                        if params['quota_matrix'][reviewer][paper_index] == 0:
                            underbids_with_coi[reviewer] = 0
                    if sum(underbids_with_coi) == 0:
                        underbids_with_coi[reviewer_index] = 1
                    fractional_allocation_profile[reviewer_index][paper_index] = \
                        min(params['quota_matrix'][reviewer_index][paper_index],
                            fractional_allocation_profile[reviewer_index][paper_index] +
                            papers_total_underbids[paper_index] * (underbids[reviewer_index] / sum(underbids_with_coi)))
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


possible_algorithms = {'FractionalAllocation': FractionalAllocation}


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