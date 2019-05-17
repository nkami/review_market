import argparse
import pathlib
import datetime
import pandas as pd
import numpy as np
from matching_algorithms import *
from evaluation import *


def write_final_results_to_csv(params, algorithm_results):
    columns = ['reviewer id', 'paper id', 'private cost', 'step 1 allocation', 'step 2 allocation', 'step 3 allocation',
               'cost contribution', 'individual contribution', 'total social cost', 'total unallocated papers',
               'total paper cost', 'matching input json file']
    total_social_cost = calculate_total_social_cost(params, algorithm_results)
    total_unallocated_papers = sum(algorithm_results['unallocated_papers'])
    individual_contribution = calculate_individual_contribution(params, algorithm_results)
    total_paper_cost = calculate_total_paper_cost(params, algorithm_results)
    data = []
    for bidder in range(-1, params['total_reviewers']):  # bidder -1 is a dummy reviewer that got the unallocated papers
        for paper in range(0, params['total_papers']):
            if bidder == -1:
                cost_contribution = (params['unallocated_papers_price'][paper] *
                                     algorithm_results['unallocated_papers'][paper])
                dummy_individual_contribution = sum([params['unallocated_papers_price'][paper] *
                                                     algorithm_results['unallocated_papers'][paper] for
                                                     paper in range(0, params['total_papers'])])
                data.append([bidder, paper, params['unallocated_papers_price'][paper], -1, -1,
                             algorithm_results['unallocated_papers'][paper], cost_contribution,
                             dummy_individual_contribution, total_social_cost, total_unallocated_papers,
                             total_paper_cost[paper], params['matching_input']])
            else:
                paper_cost = params['cost_matrix'][bidder][paper]
                if params['quota_matrix'][bidder][paper] == 0:  # in case of COI
                    paper_cost = -1
                cost_contribution = paper_cost * algorithm_results['third_step_allocation'][bidder][paper]
                data.append([bidder, paper, paper_cost, algorithm_results['first_step_allocation'][bidder][paper],
                             algorithm_results['second_step_allocation'][bidder][paper],
                             algorithm_results['third_step_allocation'][bidder][paper], cost_contribution,
                             individual_contribution[bidder], total_social_cost, total_unallocated_papers,
                             total_paper_cost[paper], params['matching_input']])
    data = np.array(data)
    data_frame = pd.DataFrame(data, columns=columns)
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    path = '.\\output\\final_allocation_{0}.csv'.format(datetime.datetime.now().isoformat()[:-7].replace(':', '-'))
    data_frame.to_csv(path, index=None, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputFile", help="a path to the input file")
    args = parser.parse_args()
    with open(args.InputFile) as file:
        params = json.loads(file.read())
    params['matching_input'] = args.InputFile
    algorithm = possible_algorithms[params['matching_algorithm']](params)
    algorithm_results = algorithm.match(params['final_bidding_profile'], params)
    write_final_results_to_csv(params, algorithm_results)
