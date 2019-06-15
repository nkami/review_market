from instance_generator import *


# Transforms a vector (a reviewer) from the cost and quota matrices into a list of tuples:
# (paper_id, paper_private_cost). A paper that has a COI with the reviewer wont appear in the returned list.
def c_q_vec_to_pairs(params, reviewer_index):
    pairs = []
    for paper in range(0, params['total_papers']):
        if params['quota_matrix'][reviewer_index][paper] != 0:  # check that no COI with paper
            pairs.append((paper, params['cost_matrix'][reviewer_index][paper]))
    return pairs


def calculate_total_social_cost(params, algorithm_results):
    social_cost = 0
    for reviewer in range(0, params['total_reviewers']):
        private_prices = c_q_vec_to_pairs(params, reviewer)
        sorted_papers_by_index = sorted(private_prices, key=lambda tup: tup[0])
        reviewer_private_prices = [pair[1] for pair in sorted_papers_by_index]
        reviewer_social_cost = [reviewer_private_prices[i] * algorithm_results['third_step_allocation'][reviewer][i]
                                for i in range(0, len(sorted_papers_by_index))]
        social_cost += sum(reviewer_social_cost)
    social_cost += sum([params['unallocated_papers_price'][paper] * algorithm_results['unallocated_papers'][paper] for
                       paper in range(0, params['total_papers'])])
    return social_cost


def calculate_individual_contribution(params, algorithm_results):
    individual_contribution = []
    for reviewer in range(0, params['total_reviewers']):
        individual_contribution.append(sum([params['cost_matrix'][reviewer][paper] *
                                            algorithm_results['third_step_allocation'][reviewer][paper] for paper in
                                            range(0, params['total_papers'])]))
    return individual_contribution


def calculate_total_paper_cost(params, algorithm_results):
    total_paper_cost = []
    for paper in range(0, params['total_papers']):
        total_paper_cost.append(sum([params['cost_matrix'][reviewer][paper] *
                                     algorithm_results['third_step_allocation'][reviewer][paper] for reviewer in
                                     range(0, params['total_reviewers'])]))
    return total_paper_cost

