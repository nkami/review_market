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


class Bidder:

    def __init__(self, params,reviewer_index,is_fallback=False):

        self.reviewer_index = reviewer_index
        self.private_costs = params['cost_matrix'][reviewer_index]
        self.price_weight = params['price_weight']
        self.paper_COI = np.array(params['quota_matrix'][self.reviewer_index]) == 0
        self.paper_thresholds = [0] * len(self.private_costs)

        if params['current_bidding_requirements'] == -1:
            k = sum(params['papers_requirements'])
            k = k / params['total_reviewers']
            self.bidding_requirement = k
        else:
            self.bidding_requirement = params['current_bidding_requirements']
        if "cost_threshold" in params:
            self.cost_threshold = params["cost_threshold"]
        if "cost_threshold2" in params:
            self.cost_threshold_strong_bid = params["cost_threshold2"]
        self.is_fallback = is_fallback
        if self.is_fallback:
            self.bidding_requirement =  params["fallback_bidding_requirement"]
        self.bidding_limit = params['bidding_limit']
        self.init(params)

    def init(self,params):
        return

    def bid(self,paper_id,current_bidding_profile):
        if self.private_costs[paper_id] <= self.cost_threshold_strong_bid:
            current_bidding_profile[self.reviewer_index][paper_id] = 2
        elif self.private_costs[paper_id] <= self.cost_threshold:
            current_bidding_profile[self.reviewer_index][paper_id] = 1

    def apply_reviewer_behavior(self, params, current_bidding_profile, prices):
        print('Method not implemented')

    def get_type(self):
        return "Bidder"


class IntegralSelectiveBidder(Bidder):
    # on initialization randomizes a price threshold for each paper as follows: papers with high cost have high chance to have a low or 0 threshold.

    def init(self, params):
        m = len(self.private_costs)
        correl = params['selective_correlation']
        fraction_sure = params['selective_fraction_sure']
        fraction_maybe = params['selective_fraction_maybe']
        fraction_price = params['selective_price_threshold']
        perturbed_costs = correl * np.array(self.private_costs)  +  (1-correl) * np.random.rand(m)
        sorted_papers_id = np.argsort(perturbed_costs)
        for idx, paper_id in enumerate(sorted_papers_id):
            if idx <= fraction_sure * m:
                self.paper_thresholds[paper_id] = 0
            elif idx <= (fraction_sure+fraction_maybe) * m:
                self.paper_thresholds[paper_id] = fraction_price
            else:
                self.paper_thresholds[paper_id] = 10


    def apply_reviewer_behavior(self, params, current_bidding_profile,prices):
        contribution = 0
        sorted_papers_id = np.argsort(self.private_costs)
        for paper_id in sorted_papers_id:
            if self.paper_COI[paper_id] == False and prices[paper_id] >= self.paper_thresholds[paper_id]:
               self.bid(paper_id,current_bidding_profile)
               contribution += prices[paper_id]
            if contribution >= self.bidding_requirement or np.sum(current_bidding_profile[self.reviewer_index] > 0) >= self.bidding_limit:
                break
        return current_bidding_profile

    def get_type(self):
        return "IntegralSelectiveBidder"

class UniformSelectiveBidder(IntegralSelectiveBidder):
    def init(self, params):
        my_params = copy.deepcopy(params)
        my_params["selective_price_threshold"] = 0
        my_params["selective_fraction_sure"] +=  params['selective_fraction_maybe']/2
        my_params['selective_fraction_maybe'] = 0
        IntegralSelectiveBidder.init(self,my_params)

    def apply_reviewer_behavior(self, params, current_bidding_profile, prices):
        my_prices = [1] * len(prices)
        IntegralSelectiveBidder.apply_reviewer_behavior(self,params, current_bidding_profile, my_prices)
        return current_bidding_profile

    def get_type(self):
        return "UniformSelectiveBidder"

class IntegralGreedyBidder(Bidder):
    # Orders papers according to (cost - price*weight) in increasing order.
    # Bids until contribution exceeds the threshold.
    # Only adds bids
    def apply_reviewer_behavior(self, params, current_bidding_profile,prices):
        contribution = 0
        priority = np.subtract(self.private_costs , np.multiply(self.price_weight, prices))
        sorted_papers_id = np.argsort(priority)
        for paper_id in sorted_papers_id:
            if self.paper_COI[paper_id] == False:
                self.bid(paper_id,current_bidding_profile)
                contribution += prices[paper_id]

            if contribution >= self.bidding_requirement or np.sum(current_bidding_profile[self.reviewer_index] > 0) >= self.bidding_limit:
                break
        return current_bidding_profile


    def get_type(self):
        return "IntegralGreedyBidder"

class IntegralSincereBidder(IntegralGreedyBidder):

    def init(self,params):
        self.price_weight = 0

    def get_type(self):
        return "IntegralSincereBidder"

class UniformBidder(IntegralSincereBidder):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere
    # integral bid until reaching the threshold
    def apply_reviewer_behavior(self, params, current_bidding_profile, prices):
        my_prices = [1]*len(prices)
        IntegralGreedyBidder.apply_reviewer_behavior(self,params, current_bidding_profile, my_prices)
        return current_bidding_profile
    def get_type(self):
        return "UniformBidder"



class BestIntegralSincereUnderbidResponse(Bidder):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a sincere
    # underbid that will yield the minimal cost value according to the private prices of the reviewer after allocation.
    def apply_reviewer_behavior(self, params, current_bidding_profile, prices):
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


class BestIntegralSincereResponse(Bidder):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a sincere bid
    # that will yield the minimal cost value according to the private prices of the reviewer after allocation.
    def apply_reviewer_behavior(self, params, current_bidding_profile, prices):
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


class BestOfXIntegralResponse(Bidder):
    # In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. The reviewer will submit a bid on X of
    # the cheapest papers (per market price) that will yield the minimal cost value according to the private prices of
    # the reviewer after allocation.
    def apply_reviewer_behavior(self, params, current_bidding_profile, prices):
        print('to do')


class BestOfXIntegralSincereResponse(Bidder):
    def apply_reviewer_behavior(self, params, current_bidding_profile, prices):
        print('to do')
