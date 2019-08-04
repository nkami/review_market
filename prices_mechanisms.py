import json
import argparse
import pandas as pd
import numpy as np
import pathlib
import datetime
from bidder_behaviors import *
from matching_algorithms import *


class Mechanism:
    def __init__(self, params):
        self.cost_matrix = params['cost_matrix']
        self.quota_matrix = params['quota_matrix']
        self.total_reviewers = params['total_reviewers']
        self.total_papers = params['total_papers']
        self.bidding_requirement = params['bidding_requirement']
        self.papers_review_requirements = params['papers_requirements']
        self.current_bidding_profile = np.zeros((params['total_reviewers'], params['total_papers']))
        self.demand = []
        self.prices = []
        self.threshold = None
        self.current_iteration = None
        self.init_demand()
        self.init_threshold()
        self.number_of_updates = 0

    def update_demand(self):
        print('Method not implemented')

    def init_demand(self):
        print('Method not implemented')

    def init_threshold(self):
        print('Method not implemented')

def price_by_demand(paper_requirement, paper_demand):
    if paper_demand <= 0:
        return 1
    else:
        return min(1, (paper_requirement / paper_demand))

class PriceMechanism(Mechanism):
    def init_demand(self):
        self.demand = []
        for paper_index in range(0, self.total_papers):
            paper_demand = np.sum(self.current_bidding_profile, axis=0)[paper_index]
            self.demand.append(paper_demand)


    def init_threshold(self):
        if isinstance(self.bidding_requirement,float) or  isinstance(self.bidding_requirement,int):
            self.threshold = self.bidding_requirement
        else:
            self.threshold = sum(self.papers_review_requirements)
            self.threshold = self.threshold / self.total_reviewers

    def update_demand(self):
        self.init_demand()

    # return prices based on the last updated demand
    def get_prices(self):
        prices = []
        for paper_index in range(0, self.total_papers):
            paper_demand = self.demand[paper_index] #= np.sum(self.current_bidding_profile, axis=0)[paper_index]
            paper_price = price_by_demand(self.papers_review_requirements[paper_index],paper_demand)
            prices.append(paper_price)
        return prices


    # updates demand only based on new_bid, then recompute price
    def get_price_for_bid(self, paper_index, bidder_index, new_bid):
        old_bid = self.current_bidding_profile[bidder_index][paper_index]
        paper_demand = self.demand[paper_index] - old_bid + new_bid
        return price_by_demand(self.papers_review_requirements[paper_index],paper_demand)


    def get_prices_for_same_bid(self,bidder_index,new_bid):
        prices = []
        for paper_index in range(0, self.total_papers):
            paper_price = self.get_price_for_bid(paper_index,bidder_index,new_bid)
            prices.append(paper_price)
        return prices



possible_mechanisms = {'PriceMechanism': PriceMechanism}
possible_behaviors = {
                      'SincereIntegralBehaviorWithMinPrice': SincereIntegralBehaviorWithMinPrice,
                      'BestIntegralSincereUnderbidResponse': BestIntegralSincereUnderbidResponse,
                      'BestIntegralSincereResponse': BestIntegralSincereResponse,
                      'IntegralSincereBehavior': IntegralSincereBehavior,
                      'IntegralGreedyBehavior': IntegralGreedyBehavior,
                      'UniformBehavior' : UniformBehavior,
                      }

# adds to output the current allocation and prices for all bidders and papers
def current_state_output(step,mec,mec_previous,algorithm_result,bidders_who_bid_since_last_update,cost_matrix,all_behavior_names,csv_name):
    n = mec.total_reviewers
    m = mec.total_papers
    current_prices = mec.get_prices()
    # TODO: this is only well-defined for the Mock algorithm. Need a more general solution.  Maybe return blank if another algorithm is used.
    unallocated_papers = np.subtract(mec.papers_review_requirements,
                                     algorithm_result['second_step_allocation'].sum(axis=0))
    final_allocation = algorithm_result['third_step_allocation']
    last_bids_data = []
    for bidder in range(0, n):
        bids = mec.current_bidding_profile[bidder]
        old_bids = mec_previous.current_bidding_profile[bidder]
        # this is the price seen by the bidder when bidding:
        # TODO: it is not computed correctly since it just considers the last price update
        seen_prices = mec_previous.get_prices_for_same_bid(bidder, 1)
        for paper in range(0, m):
            last_bids_data.append([step,
                                     mec.number_of_updates,
                                     bidder,
                                     all_behavior_names[bidder],
                                     (bidder in bidders_who_bid_since_last_update),
                                     paper,
                                     cost_matrix[bidder][paper],
                                     mec_previous.demand[paper],
                                     mec.demand[paper],
                                     current_prices[paper],
                                     seen_prices[paper],
                                     np.dot(seen_prices, bids),
                                     old_bids[paper],
                                     bids[paper],
                                     int(bids[paper] > 0),
                                     final_allocation[bidder][paper],
                                     final_allocation[bidder][paper] * cost_matrix[bidder][paper],
                                     np.sum(bids),
                                     np.dot(current_prices, bids),
                                     np.dot(cost_matrix[bidder], final_allocation[bidder]),
                                     unallocated_papers[paper],
                                     csv_name])
    return last_bids_data

def run_simulation_and_output_csv_file(params, bidding_order, time_stamp):
    market_bids_data = []
    forced_permutations = params['forced_permutations']
    num_of_steps = params['number_of_bids_until_prices_update']
    mec = possible_mechanisms[params['market_mechanism']](params)
    n = mec.total_reviewers
    # set behaviors for all reviewers:
    all_behavior_names = [params['reviewers_behavior']]*n
    all_behaviors = [] *n
    if "fallback_behavior" in params:
          fallback_probability = params["fallback_probability"]/100
          rand = np.random.rand(n)
          for i in range(n):
              if rand[i] < fallback_probability:
                  all_behavior_names[i] = params["fallback_behavior"]
    for i in range(n):
       all_behaviors.append(possible_behaviors[all_behavior_names[i]]())

    total_bids_until_closure = params['total_bids_until_closure']
    output_detail_level_permutations = params['output_detail_level_permutations']
    output_detail_level_iterations = params['output_detail_level_iterations']
    cost_matrix = params['cost_matrix']

    csv_name = 'simulation_{0}'.format(time_stamp)
    path_csv = '.\\output\\simulation_{0}.csv'.format(time_stamp)
    columns = ['#step',             # number of times that a bidder played so far
               '#updates',          # number of price updates so far (not including current)
               'reviewer id',
               'reviewer type',      # Name of reviewer behavior
               'bidder selected',     # True if this reviewer was selected to update bid since last price update
               'paper id',
               'private_cost',    # the fixed private cost of paper_id to reviewer_id
               'previous demand', # total demand of paper before update
               'new demand',      # total demand of paper before update
               'price',           # price according to new demand
               'seen price',      # price based on demand at time of bidding (after previous update)
               'total seen price', # sum of seen_price*bid
               'previous bid',     # bid amount of reviewer_id on paper_id before last update
               'bid',             # bid amount of reviewer_id on paper_id
               'positive bid',    # "1" if bid>0
               'allocation',      # allocated amount of paper_id to reviewer_id under selected algorithm (currently only supports the Mock algorithm)
               'realized cost',   # allocation*private_cost
               'total bid',       # sum of bids for this reviewer
               'total price',     # sum of bid*price for this reviewer
               'total realized cost',   # sum of realized_cost
               'unallocated_amount_step_2',  # the paper excess at the end of second allocation step (same for all bidders)
               'matching output json file']
    bidders_who_bid_since_last_update = []
    mec_before_update = copy.deepcopy(mec)
    iterations_output = 0
    permutations_output = 0
    for step, current_bidder in enumerate(bidding_order):
        update_prices = False
        output_bids = False
        mec.current_bidding_profile = all_behaviors[current_bidder].apply_reviewer_behavior(params, mec.current_bidding_profile,
                                                                                current_bidder, mec.threshold, mec.get_prices_for_same_bid(current_bidder,1))
        bidders_who_bid_since_last_update.append(current_bidder)

        #total_private_cost = sum([params['cost_matrix'][current_bidder][paper] * mec.current_bidding_profile[current_bidder][paper] *
        #                          mec.prices[paper] for paper in range(0, params['total_papers'])])
        # TODO: the logic of this part is not very clear
        if step >= n * forced_permutations:
            num_of_steps -= 1
            if num_of_steps == 0:
                update_prices = True
                if (100*(iterations_output/(1+mec.number_of_updates - forced_permutations)) < output_detail_level_iterations):
                    output_bids = True
                    iterations_output += 1
                num_of_steps = params['number_of_bids_until_prices_update']
        elif (step + 1) % n == 0:
            update_prices = True
            if 100 * (permutations_output/((step+1)/n)) < output_detail_level_permutations:
                output_bids = True
                permutations_output += 1
        # always print the last state
        if step == len(bidding_order)-1:
            output_bids = True
        if update_prices:
            print("step {0}".format(step))
            mec.update_demand()
            if output_bids:
                algorithm = possible_algorithms[params['matching_algorithm']](params)
                algorithm_result = algorithm.match(mec.current_bidding_profile, params)
                market_bids_data.extend(current_state_output(step, mec, mec_before_update,algorithm_result,bidders_who_bid_since_last_update,cost_matrix,all_behavior_names,csv_name ))
            bidders_who_bid_since_last_update = []
            mec_before_update = copy.deepcopy(mec)
            mec.number_of_updates += 1

    market_bids_data = np.array(market_bids_data)
    data_frame = pd.DataFrame(market_bids_data, columns=columns)
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    data_frame.to_csv(path_csv, index=None, header=True)
    return mec.current_bidding_profile


def output_json_file(params, final_bidding_profile, time_stamp):
    output = {'reviewers_behavior': params['reviewers_behavior'],
              'forced_permutations': params['forced_permutations'],
              'number_of_bids_until_prices_update': params['number_of_bids_until_prices_update'],
              'total_bids_until_closure': params['total_bids_until_closure'],
              'matching_algorithm': params['matching_algorithm'],
              'market_mechanism': params['market_mechanism'],
              'ignore_quota_constraints': params['ignore_quota_constraints'],
              'additional_params': {},
              'total_reviewers': params['total_reviewers'],
              'total_papers': params['total_papers'],
              'final_bidding_profile': final_bidding_profile.tolist(),
              'papers_requirements': params['papers_requirements'],
           #   'unallocated_papers_price': params['unallocated_papers_price'],
              'cost_matrix': params['cost_matrix'],
              'quota_matrix': params['quota_matrix']}
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    path_json = '.\\output\\simulation_{0}.json'.format(time_stamp)
    with open(path_json, 'w') as output_file:
        json.dump(output, output_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputFile", help="a path to the input file")
    args = parser.parse_args()
    with open(args.InputFile) as file:
        params = json.loads(file.read())
    n = params['total_reviewers']
    steps = params['total_bids_until_closure']
    bidding_order = []
    for i in range(0, params['forced_permutations']):
        bidding_order += list(range(0, n))
    # make sure everyone bids at list once before there are new bids.
    p = np.random.permutation(n).tolist()
    if steps<=n:
        bidding_order.extend(p[0:steps])
    else:
        bidding_order.extend(p)
        bidding_order.extend(np.random.randint(0, high=n,
                                       size=steps-n))
    time_stamp = datetime.datetime.now().isoformat()[:-7].replace(':', '-')
    final_bidding_profile = run_simulation_and_output_csv_file(params, bidding_order, time_stamp)
    output_json_file(params, final_bidding_profile, time_stamp)

