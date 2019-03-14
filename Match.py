import argparse
import pathlib
import datetime
import pandas as pd
import numpy as np
import random
import copy
from matching_algorithms import *
from bidder_behaviors import *
from prices_mechanisms import *
from instance_generator import *
from evaluation import *


private_prices_generators = {'SimplePrivatePricesGenerator': SimplePrivatePricesGenerator}
behaviors = {'SincereIntegralBehavior': SincereIntegralBehavior(),
             'BestTwoPreferenceBehavior': BestTwoPreferenceBehavior(),
             'BestIntegralSincereUnderbidResponse': BestIntegralSincereUnderbidResponse(),
             'BestIntegralSincereResponse': BestIntegralSincereResponse()}
mechanisms = {'FixedMechanism': FixedMechanism, 'PriceMechanism': PriceMechanism, 'MixedMechanism': MixedMechanism}
algorithms = {'FractionalAllocation': FractionalAllocation()}
possible_parameters = {'PrivatePricesGenerators': private_prices_generators, 'Algorithms': algorithms,
                       'MarketMechanisms': mechanisms, 'Behaviors': behaviors}


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("InputFile", help="a path to the input file")
    args = parser.parse_args()
    parameters = {'PreferenceFile': [], 'UnallocatedPapersPrices': [], 'PrivatePricesGenerators': [],
                  'Algorithms': [], 'MarketMechanisms': [], 'ReviewRequirements': [], 'Behaviors': []}
    file = open(args.InputFile)
    param_with_multiple_arguments = None
    for line in file.readlines():
        if ':' not in line:
            continue
        line = line.rstrip()
        current_param = line.split(': ')[0]
        arguments = line.split(': ')[1]
        if len(arguments.split(', ')) > 1:
            param_with_multiple_arguments = current_param
        if current_param in [key for key in possible_parameters]:
            parameters[current_param] = [possible_parameters[current_param][argument] for argument in
                                         arguments.split(', ')]
        else:
            parameters[current_param] = arguments.split(', ')
    for key, value in parameters.items():
        if (key == 'PreferenceFile' or key == 'PrivatePricesGenerators') and len(value) > 1:
            print('Cannot perform simulations on multiple {0} arguments.'.format(key))
            exit()
    return {'parameters': parameters, 'param_with_multiple_arguments': param_with_multiple_arguments}


def run_simulation(instance, algorithm, mechanism, behavior, bidding_order):
    mec = mechanism(instance)
    for bidding_iteration in range(0, mec.number_of_iterations):
        mec.current_iteration = bidding_iteration
        for bidder in bidding_order:
            mec.current_bidding_profile = behavior.apply_reviewer_behavior(instance, mec.current_bidding_profile,
                                                                           bidder, mec.threshold, mec.prices, algorithm)
            mec.update_prices()
    return mec.current_bidding_profile


def fit_parameters_and_run(parameters, instance, bidding_order, param_with_multiple_arguments):
    if instance == None:
        instance_generator = PreflibInstanceGenerator('.\\data\\PrefLib\\{0}'.format(parameters['PreferenceFile']),
                                                      int(parameters['ReviewRequirements']),
                                                      parameters['PrivatePricesGenerators'],
                                                      int(parameters['UnallocatedPapersPrices']))
        instance = instance_generator.generate_instance()
        bidding_order = random.sample(range(0, instance.total_reviewers), instance.total_reviewers)
        final_bid = run_simulation(instance, parameters['Algorithms'], parameters['MarketMechanisms'],
                                   parameters['Behaviors'], bidding_order)
    else:
        if param_with_multiple_arguments == 'ReviewRequirements':
            instance.papers_review_requirement = [int(parameters[param_with_multiple_arguments]) for x in
                                                  range(0, instance.total_papers)]
        if param_with_multiple_arguments == 'UnallocatedPapersPrices':
            instance.unallocated_papers_price = [int(parameters[param_with_multiple_arguments]) for x in
                                                 range(0, instance.total_papers)]
        final_bid = run_simulation(instance, parameters['Algorithms'], parameters['MarketMechanisms'],
                                   parameters['Behaviors'], bidding_order)
    return {'final_bid': final_bid, 'instance': instance, 'bidding_order': bidding_order}


def write_results_to_output(instance, algorithm_results, parameters):
    columns = ['Bidder', 'Paper', 'Private Cost', 'Step 1 Allocation', 'Step 2 Allocation', 'Step 3 Allocation',
               'Social Cost', 'Total Social Cost', 'Total Unallocated Papers', 'Preference File', 'Market Mechanism',
               'Behavior', 'Algorithm', 'Private Price Generator']
    total_social_cost = calculate_social_cost(instance, algorithm_results, int(parameters['UnallocatedPapersPrices']))
    total_unallocated_papers = sum(algorithm_results['unallocated_papers'])  # should take into account the -1 bidder
    data = []
    output_mechanism = '{0}'.format(parameters['MarketMechanisms']).split('.')[1].replace("'>", '')
    output_behavior = '{0}'.format(parameters['Behaviors']).split('.')[1].split(' object')[0]
    output_algorithms = '{0}'.format(parameters['Algorithms']).split('.')[1].split(' object')[0]
    output_private_price_generator = '{0}'.format(parameters['PrivatePricesGenerators']).split('.')[1].replace("'>", '')
    for bidder in range(-1, instance.total_reviewers):
        for paper in range(0, instance.total_papers):
            if bidder == -1:
                social_cost = instance.unallocated_papers_price[paper] * algorithm_results['unallocated_papers'][paper]
                data.append([bidder, paper, instance.unallocated_papers_price[paper], -1, -1,
                             algorithm_results['unallocated_papers'][paper], social_cost,
                             total_social_cost, total_unallocated_papers, parameters['PreferenceFile'],
                             output_mechanism, output_behavior, output_algorithms, output_private_price_generator])
            else:
                paper_cost = [pair[1] for pair in instance.private_prices[bidder] if pair[0] == paper]
                if len(paper_cost) == 0:  # in case of COI
                    paper_cost.append(-1)
                paper_cost = paper_cost[0]
                social_cost = paper_cost * algorithm_results['third_step_allocation'][bidder][paper]
                data.append([bidder, paper, paper_cost, algorithm_results['first_step_allocation'][bidder][paper],
                             algorithm_results['second_step_allocation'][bidder][paper],
                             algorithm_results['third_step_allocation'][bidder][paper], social_cost, total_social_cost,
                             total_unallocated_papers, parameters['PreferenceFile'], output_mechanism, output_behavior,
                             output_algorithms, output_private_price_generator])
    data = np.array(data)
    data_frame = pd.DataFrame(data, columns=columns)
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    path = '.\\output\\simulation_{0}.csv'.format(datetime.datetime.now().isoformat().replace(':', '_'))
    data_frame.to_csv(path, index=None, header=True)


if __name__ == '__main__':
    parameters_output = get_parameters()
    param_with_multiple_arguments = parameters_output['param_with_multiple_arguments']
    parameters = copy.deepcopy(parameters_output['parameters'])
    results = {'final_bid': None, 'instance': None, 'bidding_order': None}
    if param_with_multiple_arguments == None:  # in case only 1 simulation is runned
        param_with_multiple_arguments = 'Algorithms'
    for argument in parameters_output['parameters'][param_with_multiple_arguments]:
        parameters[param_with_multiple_arguments] = argument
        for key, value in parameters.items():
            if isinstance(value, list):
                parameters[key] = value[0]
        results = fit_parameters_and_run(parameters, results['instance'], results['bidding_order'],
                                         param_with_multiple_arguments)
        algorithm_results = parameters['Algorithms'].match(results['final_bid'], results['instance'])
        write_results_to_output(results['instance'], algorithm_results, parameters)


