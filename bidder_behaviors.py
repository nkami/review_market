import numpy as np
from instance_generator import *
from matching_algorithms import *


class BidderBehaviors:
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices):
        print('Method not implemented')


class SincereIntegralBehavior(BidderBehaviors):
    '''
    In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere
    integral bid with the lowest underbidding.
    '''
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices):
        contribution = 0
        sorted_papers_by_private_price = sorted(problem_instance.private_prices[reviewer_index], key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_price]:
            if (contribution + prices[paper]) < threshold or (contribution + prices[paper]) == threshold:
                current_bidding_profile[reviewer_index][paper] = 1
                contribution += prices[paper]
        return current_bidding_profile


class BestIntegralResponse(BidderBehaviors):  # TODO
    '''
    In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a bid that has the
    minimal cost value according to the private prices of the reviewer. We assume such bid will be an underbid and is
    sincere.
    '''
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices):
        contribution = 0
        best_response = (np.zeros(problem_instance.total_papers), np.inf)
        current_bid = (np.zeros(problem_instance.total_papers), np.inf)
        #prev_bid = current_bidding_profile[reviewer_index]
        sorted_papers_by_private_price = sorted(problem_instance.private_prices[reviewer_index], key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_price]:
            if (contribution + prices[paper]) > threshold or (contribution + prices[paper]) == threshold:
                current_bidding_profile[reviewer_index] = best_response[0]
                return current_bidding_profile
            current_bid[0][paper] = 1
            current_bidding_profile[reviewer_index] = current_bid[0]
            allocation_mats = fractional_allocation_algorithm(current_bidding_profile, problem_instance)
            cost = 0



