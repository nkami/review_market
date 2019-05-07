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
        self.papers_review_requirements = params['papers_requirements']
        self.current_bidding_profile = np.zeros((params['total_reviewers'], params['total_papers']))
        self.prices = []
        self.threshold = None
        self.current_iteration = None
        self.init_prices()
        self.init_threshold()
        self.number_of_updates = 0

    def update_prices(self):
        print('Method not implemented')

    def init_prices(self):
        print('Method not implemented')

    def init_threshold(self):
        print('Method not implemented')


class PriceMechanism(Mechanism):
    def init_prices(self):
        self.prices = []
        for paper_index in range(0, self.total_papers):
            paper_demand = 1 + np.sum(self.current_bidding_profile, axis=0)[paper_index]
            paper_price = min(1, (self.papers_review_requirements[paper_index] / paper_demand))
            self.prices.append(paper_price)

    def init_threshold(self):
        self.threshold = sum(self.papers_review_requirements)
        self.threshold = self.threshold / self.total_reviewers

    def update_prices(self):
        self.init_prices()


possible_mechanisms = {'PriceMechanism': PriceMechanism}
possible_behaviors = {'SincereIntegralBehavior': SincereIntegralBehavior,
                      'BestIntegralSincereUnderbidResponse': BestIntegralSincereUnderbidResponse,
                      'BestIntegralSincereResponse': BestIntegralSincereResponse}


def run_simulation_and_output_csv_file(params, bidding_order, time_stamp):
    market_bids_data = []
    num_of_steps = params['number_of_bids_until_prices_update']
    mec = possible_mechanisms[params['market_mechanism']](params)
    reviewer_behavior = possible_behaviors[params['reviewers_behavior']]()
    path_csv = '.\\output\\simulation_{0}.csv'.format(time_stamp)
    columns = ['#step', 'reviewer id', 'updates', 'paper id', 'private_cost', 'price', 'bid', 'total bid',
               'total price', 'total private cost', 'matching output json file']
    for step, bidder in enumerate(bidding_order):
        mec.current_bidding_profile = reviewer_behavior.apply_reviewer_behavior(params, mec.current_bidding_profile,
                                                                                bidder, mec.threshold, mec.prices)
        for paper in range(0, params['total_papers']):
            market_bids_data.append([step, bidder, mec.number_of_updates, paper,
                                     params['cost_matrix'][bidder][paper], mec.prices[paper],
                                     mec.current_bidding_profile[bidder][paper],
                                     np.sum(mec.current_bidding_profile, axis=1)[bidder],
                                     np.dot(mec.prices, mec.current_bidding_profile[bidder]),
                                     np.dot(params['cost_matrix'][bidder], mec.current_bidding_profile[bidder]),
                                     'simulation_{0}'.format(time_stamp)])
        if step >= params['total_reviewers'] * params['forced_permutations']:
            num_of_steps -= 1
            if num_of_steps == 0:
                mec.update_prices()
                mec.number_of_updates += 1
                num_of_steps = params['number_of_bids_until_prices_update']
        elif (step + 1) % params['total_reviewers'] == 0:
            mec.update_prices()
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
              'unallocated_papers_price': params['unallocated_papers_price'],
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
    bidding_order = []
    for i in range(0, params['forced_permutations']):
        bidding_order += list(range(0, params['total_reviewers']))
    bidding_order += np.random.randint(0, high=params['total_reviewers'],
                                       size=params['total_bids_until_closure']).tolist()
    time_stamp = datetime.datetime.now().isoformat()[:-7].replace(':', '-')
    final_bidding_profile = run_simulation_and_output_csv_file(params, bidding_order, time_stamp)
    output_json_file(params, final_bidding_profile, time_stamp)

