from instance_generator import *


def calculate_social_cost(problem_instance, algorithm_results, unallocated_papers_price):
    social_cost = 0
    for reviewer in range(0, problem_instance.total_reviewers):
        sorted_papers_by_index = sorted(problem_instance.private_prices[reviewer], key=lambda tup: tup[0])
        reviewer_private_prices = [pair[1] for pair in sorted_papers_by_index]
        reviewer_social_cost = [reviewer_private_prices[i] * algorithm_results['third_step_allocation'][reviewer][i]
                                for i in range(0, len(sorted_papers_by_index))]
        social_cost += sum(reviewer_social_cost)
    social_cost += unallocated_papers_price * sum(algorithm_results['unallocated_papers'])
    return social_cost

