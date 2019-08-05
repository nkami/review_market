import json
import argparse
import pandas as pd
import numpy as np
import itertools as it
import pathlib
import datetime
from bidder import *
from matching_algorithms import *


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
        demand = []
        for paper_index in range(0, self.total_papers):
            paper_demand = np.sum(self.current_bidding_profile, axis=0)[paper_index]
            demand.append(paper_demand)
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
        paper_demand = self.demand[paper_index] - old_bid + new_bid
        return self.price_by_demand(self.papers_review_requirements[paper_index], paper_demand)

    def get_prices_for_same_bid(self, bidder_index, new_bid):
        prices = []
        for paper_index in range(0, self.total_papers):
            paper_price = self.get_price_for_bid(paper_index, bidder_index, new_bid)
            prices.append(paper_price)
        return prices


possible_mechanisms = {'PriceMechanism': PriceMechanism}
possible_bidder_types = {
                      'SincereIntegralBidderWithMinPrice': SincereIntegralBidderWithMinPrice,
                      'BestIntegralSincereUnderbidResponse': BestIntegralSincereUnderbidResponse,
                      'BestIntegralSincereResponse': BestIntegralSincereResponse,
                      'IntegralSincereBidder': IntegralSincereBidder,
                      'IntegralGreedyBidder': IntegralGreedyBidder,
                      'UniformBidder': UniformBidder,
                      }

# adds to output the current allocation and prices for all bidders and papers
def current_state_output(step, mec, mec_previous, bidders, bidders_who_bid_since_last_update, input_file_name, params):
    current_prices = mec.get_prices()
    algorithm = possible_algorithms[params['matching_algorithm']](params)
    algorithm_result = algorithm.match(mec.current_bidding_profile, params)
    if params['matching_algorithm'] == 'FractionalAllocation':
        step_1_unallocated_papers = np.subtract(mec.papers_review_requirements,
                                         algorithm_result['first_step_allocation'].sum(axis=0))
        step_2_unallocated_papers = np.subtract(mec.papers_review_requirements,
                                         algorithm_result['second_step_allocation'].sum(axis=0))
    else:
        step_1_unallocated_papers = np.zeros(mec.total_papers)
        step_1_unallocated_papers[:] = -1
        step_2_unallocated_papers = step_1_unallocated_papers
    final_allocation = algorithm_result['third_step_allocation']
    last_bids_data = []
    for bidder_idx in range(0, mec.total_reviewers + 1):
        if bidder_idx != mec.total_reviewers:
            bids = mec.current_bidding_profile[bidder_idx]
            old_bids = mec_previous.current_bidding_profile[bidder_idx]
            # this is the price seen by the bidder when bidding:
            # TODO: it is not computed correctly since it just considers the last price update
            seen_prices = mec_previous.get_prices_for_same_bid(bidder_idx, 1)
            for paper in range(0, mec.total_papers):
                last_bids_data.append([step,
                                         mec.number_of_updates,
                                         bidder_idx,
                                         bidders[bidder_idx],
                                         params['current_bidding_requirements'],
                                         (bidder_idx in bidders_who_bid_since_last_update),
                                         paper,
                                         params['cost_matrix'][bidder_idx][paper],
                                         mec_previous.demand[paper],
                                         mec.demand[paper],
                                         current_prices[paper],
                                         seen_prices[paper],
                                         np.dot(seen_prices, bids),
                                         old_bids[paper],
                                         bids[paper],
                                         int(bids[paper] > 0),
                                         final_allocation[bidder_idx][paper],
                                         algorithm_result['first_step_allocation'][bidder_idx][paper],
                                         algorithm_result['second_step_allocation'][bidder_idx][paper],
                                         algorithm_result['third_step_allocation'][bidder_idx][paper],
                                         final_allocation[bidder_idx][paper] * params['cost_matrix'][bidder_idx][paper],
                                         np.sum(bids),
                                         np.dot(current_prices, bids),
                                         np.dot(params['cost_matrix'][bidder_idx], final_allocation[bidder_idx]),
                                         input_file_name])
        else:  # virtual bidder
            for paper in range(0, mec.total_papers):
                last_bids_data.append([step,
                                       mec.number_of_updates,
                                       'VB',
                                       'VB',
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
                                       step_1_unallocated_papers[paper],
                                       step_2_unallocated_papers[paper],
                                       algorithm_result['unallocated_papers'][paper],
                                       algorithm_result['unallocated_papers'][paper] * params['unallocated_papers_price'][paper],
                                       'VB',
                                       'VB',
                                       np.dot(params['unallocated_papers_price'], algorithm_result['unallocated_papers']),
                                       input_file_name])
    return last_bids_data


def run_simulation_and_output_csv_file(params, bidders, bidding_order, input_file_name, sample_path, sample_idx):
    market_bids_data = []
    mec = possible_mechanisms[params['market_mechanism']](params)
    columns = ['#step',             # number of times that a bidder played so far
               '#updates',          # number of price updates so far (not including current)
               'reviewer id',
               'reviewer type',
               'bid req',
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
               'matching input json file']
    bidders_who_bid_since_last_update = []
    mec_before_update = copy.deepcopy(mec)
    iterations_output = 0
    permutations_output = 0
    final_state = None
    num_of_steps = params['current_number_of_bids_until_prices_update']
    for step, current_bidder_idx in enumerate(bidding_order):
        update_prices = False
        output_bids = False
        current_reviewer = bidders[current_bidder_idx]

        mec.current_bidding_profile = current_reviewer.apply_reviewer_behavior(params, mec.current_bidding_profile,
                                                                               current_bidder_idx,
                                                                               mec.get_prices_for_same_bid(current_bidder_idx, 1))
        bidders_who_bid_since_last_update.append(current_bidder_idx)

        # TODO: the logic of this part is not very clear
        if step >= params['total_reviewers'] * params['current_forced_permutations']:
            num_of_steps -= 1
            if num_of_steps == 0:
                update_prices = True
                if (100*(iterations_output/(1+mec.number_of_updates - params['current_forced_permutations'])) <
                        params['output_detail_level_iterations']):
                    output_bids = True
                    iterations_output += 1
                num_of_steps = params['current_number_of_bids_until_prices_update']
        elif (step + 1) % params['total_reviewers'] == 0:
            update_prices = True
            if 100 * (permutations_output/((step+1)/params['total_reviewers'])) < params['output_detail_level_permutations']:
                output_bids = True
                permutations_output += 1
        # always print the last state
        if step == len(bidding_order) - 1:
            output_bids = True
            final_state = current_state_output(step, mec, mec_before_update, bidders, bidders_who_bid_since_last_update,
                                               input_file_name, params)
        if update_prices:
            mec.update_demand()
            if output_bids:
                market_bids_data.extend(current_state_output(step, mec, mec_before_update, bidders,
                                                             bidders_who_bid_since_last_update,
                                                             input_file_name, params))
            bidders_who_bid_since_last_update = []
            mec_before_update = copy.deepcopy(mec)
            mec.number_of_updates += 1
    if sample_idx % 100 < params['amount_of_csv_sample_outputs_per_100_samples']:
        market_bids_data = np.array(market_bids_data)
        data_frame = pd.DataFrame(market_bids_data, columns=columns)
        data_frame.to_csv(sample_path, index=None, header=True)
    final_state = np.array(final_state)
    final_state = pd.DataFrame(final_state, columns=columns)
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
#            #   'unallocated_papers_price': params['unallocated_papers_price'],
#               'cost_matrix': params['cost_matrix'],
#               'quota_matrix': params['quota_matrix']}
#     try:
#         pathlib.Path('.\\output').mkdir()
#     except FileExistsError:
#         pass
#     path_json = '.\\output\\simulation_{0}.json'.format(time_stamp)
#     with open(path_json, 'w') as output_file:
#         json.dump(output, output_file, indent=4)


def adjust_params(params):
    with open(params['cost_matrix_path']) as file:
        cost_matrix_file = json.loads(file.read())
        params['cost_matrix'] = cost_matrix_file['cost_matrix']
    with open(params['quota_matrix_path']) as file:
        quota_matrix_file = json.loads(file.read())
        params['quota_matrix'] = quota_matrix_file['quota_matrix']
    if isinstance(params['papers_requirements'], int):
        params['papers_requirements'] = [params['papers_requirements'] for i in range(0, params['total_papers'])]
    else:  # if in the future papers_requirements will be generated in a specific way per paper
        with open(params['papers_requirements']) as file:
            papers_requirements_file = json.loads(file.read())
            params['papers_requirements'] = papers_requirements_file['papers_requirements']
    if isinstance(params['unallocated_papers_price'], int):
        params['unallocated_papers_price'] = [params['unallocated_papers_price'] for i in
                                              range(0, params['total_papers'])]
    else:  # if in the future unallocated_papers_price will be generated in a specific way per paper
        with open(params['unallocated_papers_price']) as file:
            papers_requirements_file = json.loads(file.read())
            params['papers_requirements'] = papers_requirements_file['papers_requirements']
    return params


def update_current_params(params, current_parameters_combination):
    params['current_bidding_requirements'] = current_parameters_combination[0]
    params['current_forced_permutations'] = current_parameters_combination[1]
    params['current_number_of_bids_until_prices_update'] = current_parameters_combination[2]
    params['current_total_bids_until_closure'] = current_parameters_combination[3]
    params['current_fallback_probability'] = current_parameters_combination[4]
    return params


def create_bidding_order(params):
    bidding_order = []
    for i in range(0, params['current_forced_permutations']):
        bidding_order += list(range(0, params['total_reviewers']))
    # make sure everyone bids at least once before there are new bids.
    minimal_bidding = np.random.permutation(params['total_reviewers']).tolist()
    if params['current_total_bids_until_closure'] <= params['total_reviewers']:
        bidding_order.extend(minimal_bidding[0:params['current_total_bids_until_closure']])
    else:
        bidding_order.extend(minimal_bidding)
        bidding_order.extend(np.random.randint(0, high=params['total_reviewers'],
                                               size=params['current_total_bids_until_closure'] - params['total_reviewers']))
    return bidding_order


def create_bidders(params):
    bidders = []
    for i in range(0, params['total_reviewers']):
        if 'fallback_behavior' in params:
            fallback_probability = params['current_fallback_probability'] / 100
            rand = np.random.uniform(0, 1)
            if rand < fallback_probability:
                bidders.append(possible_bidder_types[params['fallback_behavior']](params))
            else:
                bidders.append(possible_bidder_types[params['reviewers_behavior']](params))
        else:
            bidders.append(possible_bidder_types[params['reviewers_behavior']](params))
    return bidders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputFile", help="a path to the input file")
    args = parser.parse_args()
    with open(args.InputFile) as file:
        params = json.loads(file.read())
    input_file_name = args.InputFile.split('\\')[-1]
    params = adjust_params(params)
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    time_stamp = datetime.datetime.now().isoformat()[:-7].replace(':', '-')
    pathlib.Path('.\\output\\simulation_{0}'.format(time_stamp)).mkdir()
    columns = ['bidding requirement',
               'forced_permutations',
               'number_of_bids_until_prices_update',
               'total_bids_until_closure',
               'fallback_probability',
               'total bids',
               'total excess papers',
               'total social cost',
               'cost matrix used',
               'quota matrix used',
               'input json file used']
    results_of_all_parameters_values = []
    all_parameters = list(it.product(params['bidding_requirements'],
                                     params['forced_permutations'],
                                     params['number_of_bids_until_prices_update'],
                                     params['total_bids_until_closure'],
                                     params['fallback_probability']))
    total_samples = params['samples'] * len(all_parameters)
    for combination_idx, current_combination in enumerate(all_parameters):
        params = update_current_params(params, current_combination)
        samples_results_of_current_combination = []
        value_folder_name = 'combination_{0}'.format(combination_idx)
        combination_path = '.\\output\\simulation_{0}\\{1}'.format(time_stamp, value_folder_name)
        pathlib.Path(combination_path).mkdir()
        for sample_idx in range(0, params['samples']):
            print('currently at sample {0} out of a total {1} samples'.format(
                combination_idx * params['samples'] + sample_idx, total_samples))
            bidders = create_bidders(params)
            bidding_order = create_bidding_order(params)
            samples_path = combination_path + '\\sample_{0}.csv'.format(sample_idx)
            final_state = run_simulation_and_output_csv_file(params, bidders, bidding_order, input_file_name,
                                                             samples_path, sample_idx)
            total_bids = sum([final_state.loc[bidder * params['total_papers'], 'total bid'] for bidder in
                              range(0, params['total_reviewers'])])
            total_social_cost = sum([final_state.loc[bidder * params['total_papers'], 'total realized cost']
                                     for bidder in range(0, params['total_reviewers'])])
            # total_excess_papers = sum([final_state.loc[bidder * params['total_papers'], 'total realized cost']
            #                            for bidder in range(0, params['total_reviewers'])])
            total_excess_papers = '?'
            results_of_all_parameters_values.append([params['current_bidding_requirements'],
                                                     params['current_forced_permutations'],
                                                     params['current_number_of_bids_until_prices_update'],
                                                     params['current_total_bids_until_closure'],
                                                     params['current_fallback_probability'],
                                                     total_bids,
                                                     total_social_cost,
                                                     total_excess_papers,
                                                     params['cost_matrix_path'],
                                                     params['quota_matrix_path'],
                                                     input_file_name])
    results_of_all_parameters_values = np.array(results_of_all_parameters_values)
    data_frame = pd.DataFrame(results_of_all_parameters_values, columns=columns)
    path_csv = '.\\output\\simulation_{0}\\all_samples.csv'.format(time_stamp)
    data_frame.to_csv(path_csv, index=None, header=True)
