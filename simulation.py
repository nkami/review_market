import json
import argparse
import pandas as pd
import numpy as np
import itertools as it
import pathlib
import datetime
import time
import os
from tqdm import tqdm
from distutils.dir_util import copy_tree
import shutil
from bidder import *
from matching_algorithms import *
from evaluation import *
from prices_mechanisms import *


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
                bidder = bidders[bidder_idx]
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
                                             0,
                                             'TBD',
                                             bids[paper],
                                             int(bids[paper] > 0),
                                             final_allocation[bidder_idx][paper],
                                             algorithm_result['first_step_allocation'][bidder_idx][paper],
                                             algorithm_result['second_step_allocation'][bidder_idx][paper],
                                             algorithm_result['third_step_allocation'][bidder_idx][paper],
                                             all_realized_costs[bidder_idx][paper],
                                             sum_bids,
                                             total_contribution[bidder_idx],
                                             sum_realized_costs[bidder_idx],
                                             algorithm_name
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
                                           0,
                                           'VB',
                                           step_1_unallocated_papers[paper],
                                           step_2_unallocated_papers[paper],
                                           algorithm_result['unallocated_papers'][paper],
                                           algorithm_result['unallocated_papers'][paper] * params['unallocated_papers_price'][paper],
                                           0,
                                           0,
                                           0,
                                           algorithm_name
                                           ])
    return last_bids_data


def determine_required_actions(step, num_of_steps_until_update, iterations_output, permutations_output, mec, params):
    update_prices = False
    output_bids = False
    if step >= params['total_reviewers'] * params['current_forced_permutations']:
        num_of_steps_until_update -= 1
        if num_of_steps_until_update == 0:
            update_prices = True
            if (100 * (iterations_output / (1 + mec.number_of_updates - params['current_forced_permutations'])) <
                    params['output_detail_level_iterations']):
                output_bids = True
                iterations_output += 1
            num_of_steps_until_update = params['current_number_of_bids_until_prices_update']
    elif (step + 1) % params["total_reviewers"] == 0:
        update_prices = True
        if 100 * (permutations_output / ((step + 1) / params["total_reviewers"])) < params['output_detail_level_permutations']:
            output_bids = True
            permutations_output += 1
    return update_prices, output_bids, num_of_steps_until_update, iterations_output, permutations_output


def start_bidding_process(params, bidders, bidding_order, sample_path, sample_idx):
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
               ]
    bidders_who_bid_since_last_update = []
    iterations_output = 0
    permutations_output = 0
    final_state = None
    if "init_balanced_bids" in params:
        init_value = params["papers_requirements"][0] / params["total_reviewers"]
        mec.current_bidding_profile = np.full((params["total_reviewers"], params["total_papers"]), init_value)
    num_of_steps = params['current_number_of_bids_until_prices_update']
    for step, current_bidder_idx in enumerate(bidding_order):
        current_reviewer = bidders[current_bidder_idx]
        mec.current_bidding_profile = current_reviewer.apply_reviewer_behavior(params, mec.current_bidding_profile,
                                                                               mec.get_prices_for_same_bid(
                                                                                   current_bidder_idx, 1))
        bidders_who_bid_since_last_update.append(current_bidder_idx)
        update_prices, output_bids, num_of_steps, iterations_output, permutations_output = determine_required_actions(step, num_of_steps, iterations_output,
                                                                                                                      permutations_output, mec, params)
        # always print the final state
        if step == len(bidding_order) - 1:
            output_bids = True
            final_state = current_state_output(step, mec, bidders, bidders_who_bid_since_last_update, params)
        if update_prices:
            mec.update_demand()
            if output_bids:
                market_bids_data.extend(current_state_output(step, mec, bidders, bidders_who_bid_since_last_update, params))
            bidders_who_bid_since_last_update = []
            mec.number_of_updates += 1
    # TODO: control this better
    if len(params["matching_algorithm"]) <= 1:
        if sample_idx % 100 < params['amount_of_csv_sample_outputs_per_100_samples']:
            market_bids_data = np.array(market_bids_data)
            data_frame = pd.DataFrame(market_bids_data, columns=columns)
            data_frame.to_csv(sample_path, index=None, header=True)
    final_state = np.array(final_state)
    final_state = pd.DataFrame(final_state, columns=columns)
    return final_state


def adjust_params(params):
    if params['random_matrices']:  # TODO: maybe merge cost matrices and quota matrices?
        cost_matrices = [file for file in os.listdir('.\\cost_matrices') if 'cost_matrix' in file]
        chosen_matrix = np.random.randint(0, high=len(cost_matrices))
        chosen_matrix = cost_matrices[chosen_matrix]
        matching_quota_matrix = [file for file in os.listdir('.\\cost_matrices') if
                                 file.replace('quota_matrix', 'cost_matrix') == chosen_matrix][0]
        matching_quota_matrix = matching_quota_matrix.replace('cost_matrix', 'quota_matrix')
        params['cost_matrix_path'] = '.\\cost_matrices\\' + chosen_matrix
        params['quota_matrix_path'] = '.\\cost_matrices\\' + matching_quota_matrix
    with open(params['cost_matrix_path']) as file:
        cost_matrix_file = json.loads(file.read())
        params['cost_matrix'] = cost_matrix_file['cost_matrix']
        params['total_reviewers'] = cost_matrix_file['total_reviewers']
        params['total_papers'] = cost_matrix_file['total_papers']
    with open(params['quota_matrix_path']) as file:
        quota_matrix_file = json.loads(file.read())
        params['quota_matrix'] = quota_matrix_file['quota_matrix']
    if isinstance(params['papers_requirements'], int):
        params['papers_requirements'] = [params['papers_requirements'] for i in range(0, params['total_papers'])]
        params['unallocated_papers_price'] = [params['unallocated_papers_price'] for i in range(0, params['total_papers'])]
    else:
        params['papers_requirements'] = [params['papers_requirements'][0] for i in range(0, params['total_papers'])]
        params['unallocated_papers_price'] = [params['unallocated_papers_price'][0] for i in
                                              range(0, params['total_papers'])]
    return params


def update_current_params(params, current_parameters_combination):
    params = adjust_params(params)
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


def add_optimal_allocation_results(params, time_stamp):  # TODO: FractionalSumOWA needs to be checked
    optimal_algorithm = FractionalSumOWA(params)
    optimal_allocation = optimal_algorithm.match(params['cost_matrix'], params)['third_step_allocation']
    optimal_allocation_columns = ['paper' + str(idx) for idx in range(0, params['total_papers'])]
    all_realized_costs = np.multiply(params['cost_matrix'], optimal_allocation)
    optimal_cost = np.sum(all_realized_costs)
    optimal_cost_column = [optimal_cost for i in range(0, params['total_reviewers'])]
    optimal_path_csv = '.\\output\\simulation_{0}\\optimal_allocation.csv'.format(time_stamp)
    optimal_data_frame = pd.DataFrame(optimal_allocation, columns=optimal_allocation_columns)
    optimal_data_frame['cost'] = optimal_cost_column
    optimal_data_frame.to_csv(optimal_path_csv, index=None, header=True)


def run_simulation(input_json, time_stamp, simulation_idx, columns):
    pd.options.mode.chained_assignment = None
    input_json_path = '.\\output\\simulation_batch_{0}\\used_input_files\\'.format(time_stamp) + input_json
    with open(input_json_path) as file:
        params = json.loads(file.read())
    pathlib.Path('.\\output\\simulation_batch_{0}\\simulation_{1}'.format(time_stamp, simulation_idx)).mkdir()
    results_of_all_parameters_values = []
    all_parameters = list(it.product(params['bidding_requirements'],
                                     params['forced_permutations'],
                                     params['number_of_bids_until_prices_update'],
                                     params['total_bids_until_closure'],
                                     params['fallback_probability'],
                                     params['price_weight']))
    total_samples = params['samples'] * len(all_parameters)
    progress_bar = tqdm(total=total_samples)
    for combination_idx, current_combination in enumerate(all_parameters):
        value_folder_name = 'combination_{0}'.format(combination_idx)
        combination_path = '.\\output\\simulation_batch_{0}\\simulation_{1}\\{2}'.format(time_stamp, simulation_idx, value_folder_name)
        pathlib.Path(combination_path).mkdir()
        for sample_idx in range(0, params['samples']):
            params = update_current_params(params, current_combination)
            cost_matrix = np.array(params['cost_matrix'])
            progress_bar.update(1)
            bidders = create_bidders(params)
            bidding_order = create_bidding_order(params)
            samples_path = combination_path + '\\sample_{0}.csv'.format(sample_idx)
            final_state = start_bidding_process(params, bidders, bidding_order, samples_path, sample_idx)
            for algorithm_name in params['matching_algorithm']:
                current_final_state = final_state.loc[final_state['matching algorithm'] == algorithm_name]
                reseted_rows = {}
                for row_number, (index, row) in enumerate(current_final_state.iterrows()):
                    reseted_rows[index] = row_number
                current_final_state.rename(index=reseted_rows, inplace=True)
                bids = np.reshape(np.array(current_final_state['positive bid']).astype(np.float),
                                  [params['total_reviewers'] + 1, params['total_papers']])
                total_bids = np.sum(bids, 0)
                realized_costs = np.array([current_final_state.loc[bidder_index * params['total_papers'], 'total realized cost'] for
                                           bidder_index in range(0, params['total_reviewers'])]).astype(np.float)
                fallback_mask = [bidder.is_fallback for bidder in bidders]
                fallback_realized_costs = realized_costs[fallback_mask]
                main_realized_costs = realized_costs[np.invert(fallback_mask)]

                # only relevant for mock algorithm (otherwise should return NaN or 0)
                step_2_allocation = np.reshape(np.array(current_final_state['step 2 allocation']).astype(np.float),
                                               [params['total_reviewers'] + 1, params['total_papers']])
                excess_papers = step_2_allocation[params['total_reviewers'], :]
                allocated_step2_papers = np.sum(step_2_allocation[0:params['total_reviewers'], :])
                step_2_realized_costs = np.multiply(step_2_allocation[0:params['total_reviewers'], :], cost_matrix)
                average_cost_per_step_2_paper = np.sum(step_2_realized_costs) / allocated_step2_papers

                total_excess_papers = excess_papers.sum()
                all_bids = np.sum(total_bids)

                results_of_all_parameters_values.append([params['total_reviewers'],
                                                         params['total_papers'],
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
                                                         input_json])
    progress_bar.close()
    results_of_all_parameters_values = np.array(results_of_all_parameters_values)
    data_frame = pd.DataFrame(results_of_all_parameters_values, columns=columns)
    # '.\\output\\simulation_batch_{0}\\simulation_{1}\\{2}'
    path_csv = '.\\output\\simulation_batch_{0}\\simulation_{1}\\all_samples.csv'.format(time_stamp, simulation_idx)
    data_frame.to_csv(path_csv, index=None, header=True)
    return results_of_all_parameters_values # maybe return a data frame instead?


if __name__ == '__main__':
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    time_stamp = datetime.datetime.now().isoformat()[:-7].replace(':', '-')
    pathlib.Path('.\\output\\simulation_batch_{0}'.format(time_stamp)).mkdir()
    parser = argparse.ArgumentParser()
    parser.add_argument("InputPath", help="a path to the input file or directory")
    args = parser.parse_args()
    copied_input_files_path = '.\\output\\simulation_batch_{0}\\used_input_files'.format(time_stamp)
    pathlib.Path(copied_input_files_path).mkdir()
    if os.path.isdir(args.InputPath):
        copy_tree(args.InputPath, copied_input_files_path)
    else:
        shutil.copy2(args.InputPath, copied_input_files_path)
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
    all_simulations_results = []
    for simulation_idx, current_json_input in enumerate(os.listdir(copied_input_files_path)):
        current_simulation_results = run_simulation(current_json_input, time_stamp, simulation_idx, columns)
        all_simulations_results.append(current_simulation_results)
        print('{0} simulations left.'.format(len(os.listdir(copied_input_files_path)) - simulation_idx - 1))