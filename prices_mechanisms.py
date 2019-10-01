import numpy as np
from bidder import *
from matching_algorithms import *
from evaluation import *


class Mechanism:
    def __init__(self, params):
        self.cost_matrix = params['cost_matrix']
        self.quota_matrix = params['quota_matrix']
        self.total_reviewers = params['total_reviewers']
        self.total_papers = params['total_papers']
        self.papers_review_requirements = params['papers_requirements']
        self.current_bidding_profile = np.zeros((params['total_reviewers'], params['total_papers']))
        self.demand = self.init_demand()
        self.number_of_updates = 0

    def update_demand(self):
        print('Method not implemented')

    def init_demand(self):
        print('Method not implemented')

    def get_prices(self):
        print('Method not implemented')


class PriceMechanism(Mechanism):
    def init_demand(self):
        capped_bids = np.minimum(self.current_bidding_profile, 1)
        demand = np.sum(capped_bids, axis=0)
        return demand

    def price_by_demand(self, paper_requirement, paper_demand):
        if paper_demand <= 0:
            return 1
        else:
            return min(1, (paper_requirement / paper_demand))

    def update_demand(self):
        self.demand = self.init_demand()

    # return prices based on the last updated demand
    def get_prices(self):
        prices = []
        for paper_index in range(0, self.total_papers):
            paper_demand = self.demand[paper_index]
            paper_price = self.price_by_demand(self.papers_review_requirements[paper_index], paper_demand)
            prices.append(paper_price)
        return prices

    # updates demand only based on new_bid, then recompute price
    def get_price_for_bid(self, paper_index, bidder_index, new_bid):
        old_bid = self.current_bidding_profile[bidder_index][paper_index]
        paper_demand = self.demand[paper_index] - min(1, old_bid) + min(1, new_bid)
        return self.price_by_demand(self.papers_review_requirements[paper_index], paper_demand)

    def get_prices_for_same_bid(self, bidder_index, new_bid):
        prices = []
        for paper_index in range(0, self.total_papers):
            paper_price = self.get_price_for_bid(paper_index, bidder_index, new_bid)
            prices.append(paper_price)
        return prices


possible_mechanisms = {'PriceMechanism': PriceMechanism}




