import numpy as np
import copy
from instance_generator import *
from matching_algorithms import *


class BidderBehaviors:
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices,
                                algorithm):
        print('Method not implemented')


class SincereIntegralBehavior(BidderBehaviors):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere
    # integral bid with the lowest underbidding.
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices,
                                algorithm):
        contribution = 0
        sorted_papers_by_private_price = sorted(problem_instance.private_prices[reviewer_index], key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_price]:
            if (contribution + prices[paper]) <= threshold:
                current_bidding_profile[reviewer_index][paper] = 1
                contribution += prices[paper]
        return current_bidding_profile


class BestTwoPreferenceBehavior(BidderBehaviors):
    # Each reviewer will bid 1 on all the papers that are in one of their best two preferences rank.
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices,
                                algorithm):
        for paper in range(0, problem_instance.total_papers):
            if (paper in problem_instance.preferences_profile[reviewer_index][0]
                    or paper in problem_instance.preferences_profile[reviewer_index][1]):
                current_bidding_profile[reviewer_index][paper] = 1
        return current_bidding_profile


class BestIntegralSincereUnderbidResponse(BidderBehaviors):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a sincere
    # underbid that will yield the minimal cost value according to the private prices of the reviewer after allocation.
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices,
                                algorithm):
        contribution = 0
        best_response = (np.zeros(problem_instance.total_papers), np.inf)
        current_bid = np.zeros(problem_instance.total_papers)
        sorted_papers_by_private_price = sorted(problem_instance.private_prices[reviewer_index], key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_price]:
            if (contribution + prices[paper]) >= threshold:
                break
            current_bid[paper] = 1
            current_bidding_profile[reviewer_index] = current_bid[0]
            final_allocation = algorithm.match(current_bidding_profile, problem_instance)['third_step_allocation']
            cost = [cost_paper[1] * final_allocation[reviewer_index][cost_paper[0]] for cost_paper in
                    problem_instance.private_prices[reviewer_index]]
            cost = sum(cost)
            if cost < best_response[1]:
                best_response = (copy.deepcopy(current_bid), cost)
            contribution += prices[paper]
        current_bidding_profile[reviewer_index] = best_response[0]
        return current_bidding_profile


class BestIntegralSincereResponse(BidderBehaviors):
    """  In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a sincere bid
    that will yield the minimal cost value according to the private prices of the reviewer after allocation."""
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices,
                                algorithm):
        best_response = (np.zeros(problem_instance.total_papers), np.inf)
        current_bid = np.zeros(problem_instance.total_papers)
        sorted_papers_by_private_price = sorted(problem_instance.private_prices[reviewer_index], key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_price]:
            current_bid[paper] = 1
            current_bidding_profile[reviewer_index] = current_bid[0]
            final_allocation = algorithm.match(current_bidding_profile, problem_instance)['third_step_allocation']
            cost = [cost_paper[1] * final_allocation[reviewer_index][cost_paper[0]] for cost_paper in
                    problem_instance.private_prices[reviewer_index]]
            cost = sum(cost)
            if cost < best_response[1]:
                best_response = (copy.deepcopy(current_bid), cost)
        current_bidding_profile[reviewer_index] = best_response[0]
        return current_bidding_profile


class BestOfXIntegralResponse(BidderBehaviors):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a bid on X of
    # the cheapest papers (per market price) that will yield the minimal cost value according to the private prices of
    # the reviewer after allocation.
    def apply_reviewer_behavior(self, problem_instance, current_bidding_profile, reviewer_index, threshold, prices,
                                algorithm):
        print('to do')
