import numpy as np
from bidder_behaviors import *
from instance_generator import *
from evaluation import *
from matching_algorithms import *


class Mechanism:
    def __init__(self, instance):
        self.problem_instance = instance
        self.current_bidding_profile = np.zeros((instance.total_reviewers, instance.total_papers))
        self.init_prices()
        self.init_threshold()

    def update_prices(self):
        print('Method not implemented')

    def init_prices(self):
        print('Method not implemented')

    def init_threshold(self):
        print('Method not implemented')


class FixedMechanism(Mechanism):
    def init_prices(self):
        self.prices = []
        for paper_index in range(0, self.problem_instance.total_papers):
            self.prices.append(1)

    def init_threshold(self):  # the thresh hold should have ceil?
        self.threshold = sum(self.problem_instance.papers_review_requirement)
        self.threshold = self.threshold / self.problem_instance.total_reviewers

    def update_prices(self):
        pass


class PriceMechanism(Mechanism):
    def init_prices(self):
        self.prices = []
        for paper_index in range(0, self.problem_instance.total_papers):
            paper_demand = 1 + np.sum(self.current_bidding_profile, axis=0)[paper_index]
            '''
            the +1 shows the price if the bidder bids the maximum amount of 1 on the paper, 
            but why should we assume that? for example why not +0.5?
            '''
            paper_price = min(1, (self.problem_instance.papers_review_requirement[paper_index] / paper_demand))
            self.prices.append(paper_price)

    def init_threshold(self):  # the threshold should have ceil?
        self.threshold = sum(self.problem_instance.papers_review_requirement)
        self.threshold = self.threshold / self.problem_instance.total_reviewers

    def update_prices(self):
        self.init_prices()















