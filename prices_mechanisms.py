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
        self.bidding_requirement = params['current_bidding_requirement']
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


class PriceMechanism(Mechanism):
    def init_demand(self):
        self.demand = []
        for paper_index in range(0, self.total_papers):
            paper_demand = np.sum(self.current_bidding_profile, axis=0)[paper_index]
            self.demand.append(paper_demand)

    def price_by_demand(self, paper_requirement, paper_demand):
        if paper_demand <= 0:
            return 1
        else:
            return min(1, (paper_requirement / paper_demand))

    def init_threshold(self):
        if isinstance(self.bidding_requirement, float) or (isinstance(self.bidding_requirement, int) and
                                                           self.bidding_requirement != -1):
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
            paper_demand = self.demand[paper_index]
            paper_price = self.price_by_demand(self.papers_review_requirements[paper_index], paper_demand)
            prices.append(paper_price)
        return prices

    # updates demand only based on new_bid, then recompute price
    def get_price_for_bid(self, paper_index, bidder_index, new_bid):
        old_bid = self.current_bidding_profile[bidder_index][paper_index]
        paper_demand = self.demand[paper_index] - old_bid + new_bid
        return self.price_by_demand(self.papers_review_requirements[paper_index], paper_demand)

    def get_prices_for_same_bid(self, bidder_index, new_bid):
        prices = []
        for paper_index in range(0, self.total_papers):
            paper_price = self.get_price_for_bid(paper_index, bidder_index, new_bid)
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
def current_state_output(step, mec, mec_previous, bidders_who_bid_since_last_update, input_file_name, params):
    current_prices = mec.get_prices()
    algorithm = possible_algorithms[params['matching_algorithm']](params)
    algorithm_result = algorithm.match(mec.current_bidding_profile, params)
    if params['matching_algorithm'] == 'FractionalAllocation':
        # TODO: this is only well-defined for the Mock algorithm. Need a more general solution.  Maybe return blank if another algorithm is used.
        unallocated_papers = np.subtract(mec.papers_review_requirements,
                                     algorithm_result['second_step_allocation'].sum(axis=0))
    else:
        unallocated_papers = np.zeros(mec.total_papers)
        unallocated_papers[:] = -1
    final_allocation = algorithm_result['third_step_allocation']
    last_bids_data = []
    for bidder in range(0, mec.total_reviewers + 1):
        if bidder != mec.total_reviewers:
            bids = mec.current_bidding_profile[bidder]
            old_bids = mec_previous.current_bidding_profile[bidder]
            # this is the price seen by the bidder when bidding:
            # TODO: it is not computed correctly since it just considers the last price update
            seen_prices = mec_previous.get_prices_for_same_bid(bidder, 1)
            for paper in range(0, mec.total_papers):
                last_bids_data.append([step,
                                         mec.number_of_updates,
                                         bidder,
                                         (bidder in bidders_who_bid_since_last_update),
                                         paper,
                                         params['cost_matrix'][bidder][paper],
                                         mec_previous.demand[paper],
                                         mec.demand[paper],
                                         current_prices[paper],
                                         seen_prices[paper],
                                         np.dot(seen_prices, bids),
                                         old_bids[paper],
                                         bids[paper],
                                         int(bids[paper] > 0),
                                         final_allocation[bidder][paper],
                                         algorithm_result['first_step_allocation'][bidder][paper],
                                         algorithm_result['second_step_allocation'][bidder][paper],
                                         algorithm_result['third_step_allocation'][bidder][paper],
                                         final_allocation[bidder][paper] * params['cost_matrix'][bidder][paper],
                                         np.sum(bids),
                                         np.dot(current_prices, bids),
                                         np.dot(params['cost_matrix'][bidder], final_allocation[bidder]),
                                         unallocated_papers[paper],
                                         input_file_name])
        else:  # virtual bidder
            for paper in range(0, mec.total_papers):
                last_bids_data.append([step,
                                       mec.number_of_updates,
                                       'VB',
                                       'VB',
                                       paper,
                                       params['unallocated_papers_price'][paper],
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       'VB',
                                       algorithm_result['unallocated_papers'][paper],
                                       algorithm_result['unallocated_papers'][paper] * params['unallocated_papers_price'][paper],
                                       'VB',
                                       'VB',
                                       np.dot(params['unallocated_papers_price'], algorithm_result['unallocated_papers']),
                                       unallocated_papers[paper],
                                       input_file_name])
    return last_bids_data

def run_simulation_and_output_csv_file(params, bidding_order, time_stamp, input_file_name, current_sample):
    market_bids_data = []
    forced_permutations = params['forced_permutations']
    num_of_steps = params['number_of_bids_until_prices_update']
    mec = possible_mechanisms[params['market_mechanism']](params)
    reviewer_behavior = possible_behaviors[params['reviewers_behavior']]()
    output_detail_level_permutations = params['output_detail_level_permutations']
    output_detail_level_iterations = params['output_detail_level_iterations']
    path_csv = '.\\output\\simulation_{0}\\sample_{1}.csv'.format(time_stamp, current_sample)
    columns = ['#step',             # number of times that a bidder played so far
               '#updates',          # number of price updates so far (not including current)
               'reviewer id',
               'bidder selected',     # True if this reviewer was selected to update bid since last price update
               'paper id',
               'private_cost',    # the fixed private cost of paper_id to reviewer_id
               'previous demand', # total demand of paper before update
               'new demand',      # total demand of paper after update
               'price',           # price according to new demand
               'seen price',      # price based on demand at time of bidding (after previous update)
               'total seen price', # sum of seen_price*bid
               'previous bid',     # bid amount of reviewer_id on paper_id before last update
               'bid',             # bid amount of reviewer_id on paper_id
               'positive bid',    # "1" if bid>0
               'allocation',      # allocated amount of paper_id to reviewer_id under selected algorithm (currently only supports the Mock algorithm)
               'step 1 allocation',
               'step 2 allocation',
               'step 3 allocation',
               'realized cost',   # allocation*private_cost
               'total bid',       # sum of bids for this reviewer
               'total price',     # sum of bid*price for this reviewer
               'total realized cost',   # sum of realized_cost
               'unallocated_amount_step_2',  # the paper excess at the end of second allocation step (same for all bidders)
               'matching input json file']
    bidders_who_bid_since_last_update = []
    mec_before_update = copy.deepcopy(mec)
    iterations_output = 0
    permutations_output = 0
    final_state = None
    for step, current_bidder in enumerate(bidding_order):
        update_prices = False
        output_bids = False
        mec.current_bidding_profile = reviewer_behavior.apply_reviewer_behavior(params, mec.current_bidding_profile,
                                                                                current_bidder, mec.threshold,
                                                                                mec.get_prices_for_same_bid(current_bidder, 1))
        bidders_who_bid_since_last_update.append(current_bidder)

        #total_private_cost = sum([params['cost_matrix'][current_bidder][paper] * mec.current_bidding_profile[current_bidder][paper] *
        #                          mec.prices[paper] for paper in range(0, params['total_papers'])])
        # TODO: the logic of this part is not very clear
        if step >= params['total_reviewers'] * forced_permutations:
            num_of_steps -= 1
            if num_of_steps == 0:
                update_prices = True
                if (100*(iterations_output/(1+mec.number_of_updates - forced_permutations)) < output_detail_level_iterations):
                    output_bids = True
                    iterations_output += 1
                num_of_steps = params['number_of_bids_until_prices_update']
        elif (step + 1) % params['total_reviewers'] == 0:
            update_prices = True
            if 100 * (permutations_output/((step+1)/params['total_reviewers'])) < output_detail_level_permutations:
                output_bids = True
                permutations_output += 1
        # always print the last state
        if step == len(bidding_order) - 1:
            output_bids = True
            final_state = current_state_output(step, mec, mec_before_update, bidders_who_bid_since_last_update,
                                               input_file_name, params)
        if update_prices:
            mec.update_demand()
            if output_bids:
                market_bids_data.extend(current_state_output(step, mec, mec_before_update,
                                                             bidders_who_bid_since_last_update,
                                                             input_file_name, params))
            bidders_who_bid_since_last_update = []
            mec_before_update = copy.deepcopy(mec)
            mec.number_of_updates += 1
    market_bids_data = np.array(market_bids_data)
    data_frame = pd.DataFrame(market_bids_data, columns=columns)
    data_frame.to_csv(path_csv, index=None, header=True)
    return final_state


# def output_json_file(params, final_bidding_profile, time_stamp):
#     output = {'reviewers_behavior': params['reviewers_behavior'],
#               'forced_permutations': params['forced_permutations'],
#               'number_of_bids_until_prices_update': params['number_of_bids_until_prices_update'],
#               'total_bids_until_closure': params['total_bids_until_closure'],
#               'matching_algorithm': params['matching_algorithm'],
#               'market_mechanism': params['market_mechanism'],
#               'ignore_quota_constraints': params['ignore_quota_constraints'],
#               'additional_params': {},
#               'total_reviewers': params['total_reviewers'],
#               'total_papers': params['total_papers'],
#               'final_bidding_profile': final_bidding_profile.tolist(),
#               'papers_requirements': params['papers_requirements'],
#               'unallocated_papers_price': params['unallocated_papers_price'],
#               'cost_matrix': params['cost_matrix'],
#               'quota_matrix': params['quota_matrix']}
#     try:
#         pathlib.Path('.\\output').mkdir()
#     except FileExistsError:
#         pass
#     path_json = '.\\output\\simulation_{0}.json'.format(time_stamp)
#     with open(path_json, 'w') as output_file:
#         json.dump(output, output_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputFile", help="a path to the input file")
    args = parser.parse_args()
    with open(args.InputFile) as file:
        params = json.loads(file.read())
    input_file_name = args.InputFile.split('\\')[-1]
    steps = params['total_bids_until_closure']
    samples = params['samples']
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    columns = ['bidding requirement',
               'samples',
               'behavior',
               'total bids',
               'total excess papers',
               'total social cost',
               'matching input json']
    results_of_all_bid_req_values = []
    for current_bidding_requirement in range(0, len(params['bidding_requirements'])):
        params['current_bidding_requirement'] = params['bidding_requirements'][current_bidding_requirement]
        time_stamp = datetime.datetime.now().isoformat()[:-7].replace(':', '-')
        pathlib.Path('.\\output\\simulation_{0}'.format(time_stamp)).mkdir()
        samples_results_of_current_bid_req = []
        for current_sample in range(0, samples):
            bidding_order = []
            for i in range(0, params['forced_permutations']):
                bidding_order += list(range(0, params['total_reviewers']))
            # make sure everyone bids at least once before there are new bids.
            minimal_bidding = np.random.permutation(params['total_reviewers']).tolist()
            if params['total_bids_until_closure'] <= params['total_reviewers']:
                bidding_order.extend(minimal_bidding[0:params['total_bids_until_closure']])
            else:
                bidding_order.extend(minimal_bidding)
                bidding_order.extend(np.random.randint(0, high=params['total_reviewers'],
                                                       size=params['total_bids_until_closure'] - params['total_reviewers']))
            final_state = run_simulation_and_output_csv_file(params, bidding_order, time_stamp, input_file_name,
                                                             current_sample)
            total_bids = sum([final_state[bidder * params['total_papers']][19] for bidder in
                              range(0, params['total_reviewers'])])
            total_social_cost = sum([final_state[bidder * params['total_papers']][21] for bidder in
                                     range(0, params['total_reviewers'])])
            total_excess_papers = 0
            for idx in range(0, params['total_reviewers'] * params['total_papers']):
                total_excess_papers += final_state[idx][22]
            samples_results_of_current_bid_req.append((total_bids, total_social_cost, total_excess_papers))
        averaged_bids = sum([samples_results_of_current_bid_req[i][0] for i in range(0, samples)]) / samples
        averaged_social_cost = sum([samples_results_of_current_bid_req[i][1] for i in range(0, samples)]) / samples
        averaged_excess_papers = sum([samples_results_of_current_bid_req[i][2] for i in range(0, samples)]) / samples
        results_of_all_bid_req_values.append([params['bidding_requirements'][current_bidding_requirement],
                                              samples,
                                              params['reviewers_behavior'],
                                              averaged_bids,
                                              averaged_excess_papers,
                                              averaged_social_cost,
                                              input_file_name])
    time_stamp = datetime.datetime.now().isoformat()[:-7].replace(':', '-')
    path_csv = '.\\output\\simulation_{0}.csv'.format(time_stamp)
    results_of_all_bid_req_values = np.array(results_of_all_bid_req_values)
    data_frame = pd.DataFrame(results_of_all_bid_req_values, columns=columns)
    data_frame.to_csv(path_csv, index=None, header=True)
