import json
import argparse
import pandas as pd
import numpy as np
import itertools as it
import pathlib
import datetime
import time
from tqdm import tqdm
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
                      'IntegralSelectiveBidder' : IntegralSelectiveBidder,
                      'UniformSelectiveBidder': UniformSelectiveBidder,
                      'BestIntegralSincereUnderbidResponse': BestIntegralSincereUnderbidResponse,
                      'BestIntegralSincereResponse': BestIntegralSincereResponse,
                      'IntegralSincereBidder': IntegralSincereBidder,
                      'IntegralGreedyBidder': IntegralGreedyBidder,
                      'UniformBidder': UniformBidder,
                      }
# Gini and Hoover formulas taken from http://www.nickmattei.net/docs/papers.pdf
def gini_index(u):
    um = np.matrix(u)
    U = np.abs(np.transpose(um) - um)
    return U.sum() / (2*len(u)*um.sum())


def hoover_index(u):
    us = np.sum(u)
    return np.sum(np.abs(u - us/len(u))) / (2*us)


# adds to output the current allocation and prices for all bidders and papers
def current_state_output(step, mec, bidders, bidders_who_bid_since_last_update, params):
    current_prices = mec.get_prices()
    last_bids_data = []
    for algorithm_name in params['matching_algorithm']:
        algorithm = possible_algorithms[algorithm_name](params)
        algorithm_result = algorithm.match(mec.current_bidding_profile, params)
        if algorithm_name == 'FractionalAllocation':
            step_1_unallocated_papers = np.subtract(mec.papers_review_requirements,
                                             algorithm_result['first_step_allocation'].sum(axis=0))
            step_2_unallocated_papers = np.subtract(mec.papers_review_requirements,
                                             algorithm_result['second_step_allocation'].sum(axis=0))
        else:
            step_1_unallocated_papers = [-1 for i in range(0, mec.total_papers)]
            step_2_unallocated_papers = step_1_unallocated_papers
        final_allocation = algorithm_result['third_step_allocation']
        total_contribution = np.dot(mec.current_bidding_profile, current_prices)
        all_realized_costs = np.multiply(params['cost_matrix'], final_allocation)
        sum_realized_costs = np.sum(all_realized_costs, 1)
        for bidder_idx in range(0, mec.total_reviewers + 1):
            if bidder_idx != mec.total_reviewers:
                bids = mec.current_bidding_profile[bidder_idx]
                sum_bids = np.sum(bids)
                pos_bids = (bids > 0).astype(int)
                bidder = bidders[bidder_idx]
                old_bids = bids
                # this is the price seen by the bidder when bidding:
                # TODO: it is not computed correctly since it just considers the last price update
                # seen_prices = mec_previous.get_prices_for_same_bid(bidder_idx, 1)
                for paper in range(0, mec.total_papers):
                    last_bids_data.append([step,
                                             mec.number_of_updates,
                                             bidder_idx,
                                             bidder.get_type(),
                                             bidder.bidding_requirement,
                                             (bidder_idx in bidders_who_bid_since_last_update),
                                             paper,
                                             bidder.private_costs[paper],
                                             'TBD',
                                             mec.demand[paper],
                                             current_prices[paper],
                                             'TBD',
                                             0,#np.dot(seen_prices, bids),
                                             'TBD',
                                             bids[paper],
                                             int(bids[paper] > 0),
                                             final_allocation[bidder_idx][paper],
                                             algorithm_result['first_step_allocation'][bidder_idx][paper],
                                             algorithm_result['second_step_allocation'][bidder_idx][paper],
                                             algorithm_result['third_step_allocation'][bidder_idx][paper],
                                             #final_allocation[bidder_idx][paper] * bidder.private_costs[paper],
                                             all_realized_costs[bidder_idx][paper],
                                             sum_bids,
                                             total_contribution[bidder_idx],
                                             sum_realized_costs[bidder_idx],
                                             algorithm_name
                                             #np.dot(params['cost_matrix'][bidder_idx], final_allocation[bidder_idx]),
                                           #  input_file_name
                                           ])
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
                                           0,
                                           'VB',
                                           'VB',
                                           step_1_unallocated_papers[paper],
                                           step_2_unallocated_papers[paper],
                                           algorithm_result['unallocated_papers'][paper],
                                           algorithm_result['unallocated_papers'][paper] * params['unallocated_papers_price'][paper],
                                           0,
                                           0,
                                           0,#np.dot(params['unallocated_papers_price'], algorithm_result['unallocated_papers']),
                                           #input_file_name
                                           algorithm_name
                                           ])
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
               'matching algorithm'
               #'matching input json file'
               ]
    bidders_who_bid_since_last_update = []
    #mec_before_update = copy.deepcopy(mec)
    iterations_output = 0
    permutations_output = 0
    final_state = None
    num_of_steps = params['current_number_of_bids_until_prices_update']
    for step, current_bidder_idx in enumerate(bidding_order):
        update_prices = False
        output_bids = False
        current_reviewer = bidders[current_bidder_idx]

        mec.current_bidding_profile = current_reviewer.apply_reviewer_behavior(params, mec.current_bidding_profile,
                                                                               mec.get_prices_for_same_bid(
                                                                                   current_bidder_idx, 1))
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
            final_state = current_state_output(step, mec, bidders, bidders_who_bid_since_last_update, params)
        if update_prices:
            mec.update_demand()
            if output_bids:
                market_bids_data.extend(current_state_output(step, mec, bidders, bidders_who_bid_since_last_update,
                                                             params))
            bidders_who_bid_since_last_update = []
            #mec_before_update = copy.deepcopy(mec)
            mec.number_of_updates += 1
    if sample_idx % 100 < params['amount_of_csv_sample_outputs_per_100_samples']:
        market_bids_data = np.array(market_bids_data)
        data_frame = pd.DataFrame(market_bids_data, columns=columns)
        data_frame.to_csv(sample_path, index=None, header=True)
   # final_state = np.array(final_state)
    final_state = pd.DataFrame(final_state, columns=columns)
    return final_state


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
    params['current_price_weight'] = current_parameters_combination[5]
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
                bidders.append(possible_bidder_types[params['fallback_behavior']](params, i, True))
            else:
                bidders.append(possible_bidder_types[params['reviewers_behavior']](params, i))
        else:
            bidders.append(possible_bidder_types[params['reviewers_behavior']](params, i))
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
    columns = ['n',
               'm',
               'bidding requirement',
               'forced permutations',
               'number of bids until prices update',
               'total bids until closure',
               'fallback probability',
               'price weight',
               'matching algorithm',
               'sample index',
               'total bids',
               'total_excess_papers',
               'allocated_papers_per_bid',
               'gini_paper_bids',
               'hoover_paper_bids',
               'average_cost_per_step_2_paper',
               'average_bidder_cost',
               'gini_bidder_cost',
               'hoover_bidder_cost',
               'average_fallback_bidder_cost',
               'gini_fallback_bidder_cost',
               'hoover_fallback_bidder_cost',
               'average_main_bidder_cost',
               'gini_main_bidder_cost',
               'hoover_main_bidder_cost',
               'cost matrix used',
               'quota matrix used',
               'input json file used']
    results_of_all_parameters_values = []
    all_parameters = list(it.product(params['bidding_requirements'],
                                     params['forced_permutations'],
                                     params['number_of_bids_until_prices_update'],
                                     params['total_bids_until_closure'],
                                     params['fallback_probability'],
                                     params['price_weight']))
    total_samples = params['samples'] * len(all_parameters)
    progress_bar = tqdm(total=total_samples)
    # TODO: need to change of cost matrix changes across samples
    cost_matrix = np.array(params['cost_matrix'])
    for combination_idx, current_combination in enumerate(all_parameters):
        params = update_current_params(params, current_combination)
        m = params['total_papers']
        n = params['total_reviewers']
        samples_results_of_current_combination = []
        value_folder_name = 'combination_{0}'.format(combination_idx)
        combination_path = '.\\output\\simulation_{0}\\{1}'.format(time_stamp, value_folder_name)
        pathlib.Path(combination_path).mkdir()
        for sample_idx in range(0, params['samples']):
            progress_bar.update(1)
            bidders = create_bidders(params)
            bidding_order = create_bidding_order(params)
            samples_path = combination_path + '\\sample_{0}.csv'.format(sample_idx)
            final_state = run_simulation_and_output_csv_file(params, bidders, bidding_order, input_file_name,
                                                             samples_path, sample_idx)
            for algorithm_name in params['matching_algorithm']:
                current_final_state = final_state.loc[final_state['matching algorithm'] == algorithm_name]
                reseted_rows = {}
                for row_number, (index, row) in enumerate(current_final_state.iterrows()):
                    reseted_rows[index] = row_number
                current_final_state.rename(index=reseted_rows, inplace=True)
                # TODO: maybe keep in array format from the beginning
                bids = np.reshape(np.array(current_final_state['bid']), [n+1, m])
                total_bids = np.sum(bids, 0)
                realized_costs = np.array([current_final_state.loc[bidder_index * m, 'total realized cost'] for
                                           bidder_index in range(0, n)])
                fallback_mask = [bidder.is_fallback for bidder in bidders]
                fallback_realized_costs = realized_costs[fallback_mask]
                main_realized_costs = realized_costs[np.invert(fallback_mask)]

                # only relevant for mock algorithm (otherwise should return NaN or 0)
                step_2_allocation = np.reshape(np.array(current_final_state['step 2 allocation']), [n+1, m])
                excess_papers = step_2_allocation[n, :]
                allocated_step2_papers = np.sum(step_2_allocation[0:n, :])
                step_2_realized_costs = np.multiply(step_2_allocation[0:n, :], cost_matrix)
                average_cost_per_step_2_paper = np.sum(step_2_realized_costs) / allocated_step2_papers

                total_excess_papers = excess_papers.sum()
                all_bids = np.sum(total_bids)

                results_of_all_parameters_values.append([n,
                                                         m,
                                                         params['current_bidding_requirements'],
                                                         params['current_forced_permutations'],
                                                         params['current_number_of_bids_until_prices_update'],
                                                         params['current_total_bids_until_closure'],
                                                         params['current_fallback_probability'],
                                                         params['current_price_weight'],
                                                         algorithm_name,
                                                         sample_idx,
                                                         all_bids,
                                                         total_excess_papers,
                                                         allocated_step2_papers / all_bids,
                                                         gini_index(total_bids),
                                                         hoover_index(total_bids),
                                                         average_cost_per_step_2_paper,
                                                         np.mean(realized_costs),
                                                         gini_index(realized_costs),
                                                         hoover_index(realized_costs),
                                                         np.mean(fallback_realized_costs),
                                                         gini_index(fallback_realized_costs),
                                                         hoover_index(fallback_realized_costs),
                                                         np.mean(main_realized_costs),
                                                         gini_index(main_realized_costs),
                                                         hoover_index(main_realized_costs),
                                                         params['cost_matrix_path'],
                                                         params['quota_matrix_path'],
                                                         input_file_name])
    progress_bar.close()
    results_of_all_parameters_values = np.array(results_of_all_parameters_values)
    data_frame = pd.DataFrame(results_of_all_parameters_values, columns=columns)
    path_csv = '.\\output\\simulation_{0}\\all_samples.csv'.format(time_stamp)
    data_frame.to_csv(path_csv, index=None, header=True)
