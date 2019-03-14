import numpy as np
from bidder_behaviors import *
from instance_generator import *
from evaluation import *
from matching_algorithms import *


class Mechanism:
    def __init__(self, instance):
        self.problem_instance = instance
        self.current_bidding_profile = np.zeros((instance.total_reviewers, instance.total_papers))
        self.prices = []
        self.threshold = None
        self.current_iteration = None
        self.init_prices()
        self.init_threshold()

    def update_prices(self):
        print('Method not implemented')

    def init_prices(self):
        print('Method not implemented')

    def init_threshold(self):
        print('Method not implemented')


class FixedMechanism(Mechanism):
    def __init__(self, instance):
        super().__init__(instance)
        self.number_of_iterations = 1

    def init_prices(self):
        self.prices = []
        for paper_index in range(0, self.problem_instance.total_papers):
            self.prices.append(1)

    def init_threshold(self):
        self.threshold = sum(self.problem_instance.papers_review_requirement)
        self.threshold = self.threshold / self.problem_instance.total_reviewers

    def update_prices(self):
        pass


class PriceMechanism(Mechanism):
    def __init__(self, instance):
        super().__init__(instance)
        self.number_of_iterations = 1

    def init_prices(self):
        self.prices = []
        for paper_index in range(0, self.problem_instance.total_papers):
            paper_demand = 1 + np.sum(self.current_bidding_profile, axis=0)[paper_index]
            paper_price = min(1, (self.problem_instance.papers_review_requirement[paper_index] / paper_demand))
            self.prices.append(paper_price)

    def init_threshold(self):
        self.threshold = sum(self.problem_instance.papers_review_requirement)
        self.threshold = self.threshold / self.problem_instance.total_reviewers

    def update_prices(self):
        self.init_prices()


class MixedMechanism(Mechanism):
    def __init__(self, instance):
        super().__init__(instance)
        self.number_of_iterations = 2

    def init_prices(self):
        self.prices = []
        for paper_index in range(0, self.problem_instance.total_papers):
            paper_demand = 1 + np.sum(self.current_bidding_profile, axis=0)[paper_index]
            paper_price = min(1, (self.problem_instance.papers_review_requirement[paper_index] / paper_demand))
            self.prices.append(paper_price)

    def init_threshold(self):
        self.threshold = sum(self.problem_instance.papers_review_requirement)
        self.threshold = self.threshold / self.problem_instance.total_reviewers

    def update_prices(self):
        if self.current_iteration == 0:
            pass
        elif self.current_iteration == 1:
            self.init_prices()












