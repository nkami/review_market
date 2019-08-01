import numpy as np
import copy
from matching_algorithms import *

# TODO: there should be a class for "reviewer" rather than "behavior". This class also contains all relevant parameters...
#  including thresholds and each reviewer may have a different type. the apply_reviwer_behavior() should be....
#  in this class, and only get current profile and prices as parameters.
# Transforms a vector (a reviewer) from the cost and quota matrices into a list of tuples:
# (paper_id, paper_private_cost). A paper that has a COI with the reviewer wont appear in the returned list.
def c_q_vec_to_pairs(params, reviewer_index):
    pairs = []
    for paper in range(0, params['total_papers']):
        if params['quota_matrix'][reviewer_index][paper] != 0:  # check that no COI with paper
            pairs.append((paper, params['cost_matrix'][reviewer_index][paper]))
    return pairs


class BidderBehaviors:
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        print('Method not implemented')




class SincereIntegralBehaviorWithMinPrice(BidderBehaviors):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere
    # integral bid with the lowest underbidding.
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        contribution = 0
        min_price = params['min_price']
        private_costs = c_q_vec_to_pairs(params, reviewer_index)
        sorted_papers_by_private_cost = sorted(private_costs, key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_cost]:
            if (prices[paper] >= min_price or current_bidding_profile[reviewer_index][paper] == 1) :
                current_bidding_profile[reviewer_index][paper] = 1
                contribution += prices[paper]
            else:
                current_bidding_profile[reviewer_index][paper] = 0
            # only adds papers whose price is above min_price, but does not remove them otherwise
            if contribution >= threshold:
                break
        return current_bidding_profile

class IntegralGreedyBehavior(BidderBehaviors):
    # Orders papers according to (cost - price*weight) in increasing order.
    # Bids until contribution exceeds the threshold.
    # Only adds bids
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        contribution = 0
        if "cost_threshold" in params:
            cost_threshold = params['cost_threshold']
        else:
            cost_threshold = 100
        if "cost_threshold2" in params:
            cost_threshold2 = params['cost_threshold2']
        else:
            cost_threshold2 = 0
        if "price_weight" in params:
            price_weight = params['price_weight']
        else:
            price_weight = 0
        private_costs = params['cost_matrix'][reviewer_index]
        paper_COI = np.array(params['quota_matrix'][reviewer_index])==0
        priority = np.subtract(private_costs , np.multiply(price_weight, prices))
        sorted_papers_id = np.argsort(priority)
        for paper_id in sorted_papers_id:
            if paper_COI[paper_id] == False:
                if private_costs[paper_id] <= cost_threshold2:
                    current_bidding_profile[reviewer_index][paper_id] = 2
                    contribution += prices[paper_id]
                elif private_costs[paper_id] <= cost_threshold:
                    current_bidding_profile[reviewer_index][paper_id] = 1
                    contribution += prices[paper_id]

            if contribution >= threshold:
                break
        return current_bidding_profile

class IntegralSincereBehavior(IntegralGreedyBehavior):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere
    # integral bid until reaching the threshold
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        # contribution = 0
        # private_costs = c_q_vec_to_pairs(params, reviewer_index)
        # sorted_papers_by_private_cost = sorted(private_costs, key=lambda tup: tup[1])
        # for paper in [pair[0] for pair in sorted_papers_by_private_cost]:
        #     current_bidding_profile[reviewer_index][paper] = 0
        #     if (contribution + prices[paper]) <= threshold:
        #         current_bidding_profile[reviewer_index][paper] = 1
        #         contribution += prices[paper]
        #     else:
        #         break
        my_params = copy.deepcopy(params)
        my_params['price_weight'] = 0
        IntegralGreedyBehavior.apply_reviewer_behavior(self,my_params,current_bidding_profile,reviewer_index,threshold,prices)
        return current_bidding_profile

class UniformBehavior(IntegralSincereBehavior):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere
    # integral bid until reaching the threshold
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        my_prices = [1]*len(prices)
        IntegralGreedyBehavior.apply_reviewer_behavior(self,params,current_bidding_profile,reviewer_index,threshold,my_prices)
        return current_bidding_profile

# class FixedBehaviorCostThreshold(BidderBehaviors):
#     # bids on all papers with cost below a given threshold
#     def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
#         if "cost_threshold" in params:
#             cost_threshold = params['cost_threshold']
#         else:
#             cost_threshold = 100
#         if "cost_threshold2" in params:
#             cost_threshold2 = params['cost_threshold2']
#         else:
#             cost_threshold2 = 0
#         private_costs = params['cost_matrix'][reviewer_index]
#         paper_COI = np.array(params['quota_matrix'][reviewer_index])==0
#         m = len(private_costs)
#         for paper_id in range(m):
#             if paper_COI[paper_id] == False:
#                 if private_costs[paper_id] <= cost_threshold2:
#                     current_bidding_profile[reviewer_index][paper_id] = 2
#                 elif private_costs[paper_id] <= cost_threshold:
#                     current_bidding_profile[reviewer_index][paper_id] = 1
#         return current_bidding_profile

# class BestTwoPreferenceBehavior(BidderBehaviors):
#     # Each reviewer will bid 1 on all the papers that are in one of their best two preferences rank.
#     def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
#         for paper in range(0, problem_instance.total_papers):
#             if (paper in problem_instance.preferences_profile[reviewer_index][0]
#                     or paper in problem_instance.preferences_profile[reviewer_index][1]):
#                 current_bidding_profile[reviewer_index][paper] = 1
#         return current_bidding_profile


class BestIntegralSincereUnderbidResponse(BidderBehaviors):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a sincere
    # underbid that will yield the minimal cost value according to the private prices of the reviewer after allocation.
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        contribution = 0
        best_response = (np.zeros(params['total_papers']), np.inf)
        current_bid = np.zeros(params['total_papers'])
        private_prices = c_q_vec_to_pairs(params, reviewer_index)
        sorted_papers_by_private_price = sorted(private_prices, key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_price]:
            if (contribution + prices[paper]) >= threshold:
                break
            current_bid[paper] = 1
            current_bidding_profile[reviewer_index] = current_bid
            algorithm = possible_algorithms[params['matching_algorithm']](params)
            final_allocation = algorithm.match(current_bidding_profile, params)['third_step_allocation']
            private_prices = c_q_vec_to_pairs(params, reviewer_index)
            cost = [pair[1] * final_allocation[reviewer_index][pair[0]] for pair in private_prices]
            cost = sum(cost)
            if cost < best_response[1]:
                best_response = (copy.deepcopy(current_bid), cost)
            contribution += prices[paper]
        current_bidding_profile[reviewer_index] = best_response[0]
        return current_bidding_profile


class BestIntegralSincereResponse(BidderBehaviors):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a sincere bid
    # that will yield the minimal cost value according to the private prices of the reviewer after allocation.
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        best_response = (np.zeros(params['total_papers']), np.inf)
        current_bid = np.zeros(params['total_papers'])
        private_prices = c_q_vec_to_pairs(params, reviewer_index)
        sorted_papers_by_private_price = sorted(private_prices, key=lambda tup: tup[1])
        for paper in [pair[0] for pair in sorted_papers_by_private_price]:
            current_bid[paper] = 1
            current_bidding_profile[reviewer_index] = current_bid
            algorithm = possible_algorithms[params['matching_algorithm']](params)
            final_allocation = algorithm.match(current_bidding_profile, params)['third_step_allocation']
            private_prices = c_q_vec_to_pairs(params, reviewer_index)
            cost = [pair[1] * final_allocation[reviewer_index][pair[0]] for pair in private_prices]
            cost = sum(cost)
            if cost < best_response[1]:
                best_response = (copy.deepcopy(current_bid), cost)
        current_bidding_profile[reviewer_index] = best_response[0]
        return current_bidding_profile


class BestOfXIntegralResponse(BidderBehaviors):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a bid on X of
    # the cheapest papers (per market price) that will yield the minimal cost value according to the private prices of
    # the reviewer after allocation.
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        print('to do')


class BestOfXIntegralSincereResponse(BidderBehaviors):
    def apply_reviewer_behavior(self, params, current_bidding_profile, reviewer_index, threshold, prices):
        print('to do')
