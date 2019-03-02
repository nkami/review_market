import numpy as np
import copy

'''
you can find a detailed explanation of this algorithm in the review_market.new2 pdf, pages 5-7
'''

# what should be done with COI?
def fractional_allocation_algorithm(bidding_profile, problem_instance):
    total_reviewers = problem_instance.total_reviewers
    total_papers = problem_instance.total_papers
    # step I
    prices = []
    for paper_index in range(0, total_papers):
        paper_demand = np.sum(bidding_profile, axis=0)[paper_index]
        # what do you do if the demand is 0?
        if paper_demand == 0:
            paper_price = 1
        else:
            paper_price = min(1, (problem_instance.papers_review_requirement[paper_index] / paper_demand))
        prices.append(paper_price)
    fractional_allocation_profile = np.zeros((total_reviewers, total_papers))
    for reviewer_index in range(0, total_reviewers):
        for paper_index in range(0, total_papers):
            fractional_allocation_profile[reviewer_index][paper_index] = (bidding_profile[reviewer_index][paper_index] *
                                                                          prices[paper_index])
    first_step_allocation = copy.deepcopy(fractional_allocation_profile)
    # step II
    overbidders = []
    underbids = []
    # should k have ceil?
    k = sum(problem_instance.papers_review_requirement)
    k = k / total_reviewers #np.ceil(k / total_reviewers)
    for reviewer_index in range(0, total_reviewers):
        overbid_of_reviewer = np.sum(fractional_allocation_profile, axis=1)[reviewer_index] - k
        if overbid_of_reviewer > 0:
            overbidders.append((reviewer_index, overbid_of_reviewer))
            underbids.append(0)
        else:
            underbids.append(abs(overbid_of_reviewer))
    for overbidder in overbidders:
        for paper_index in range(0, total_papers):
            fractional_allocation_profile[overbidder[0]][paper_index] *= (k / (k + overbidder[1]))
    second_step_allocation = copy.deepcopy(fractional_allocation_profile)
    # step III
    # what if no one under bids? e.g sum(underbids) = 0?
    papers_total_underbids = []
    for paper_index in range(0, total_papers):
        paper_total_underbid = (problem_instance.papers_review_requirement[paper_index]
                                - np.sum(fractional_allocation_profile, axis=0)[paper_index])
        papers_total_underbids.append(paper_total_underbid)
    for reviewer_index in range(0, total_reviewers):
        for paper_index in range(0, total_papers):
            fractional_allocation_profile[reviewer_index][paper_index] = \
                min(1, fractional_allocation_profile[reviewer_index][paper_index] +
                    papers_total_underbids[paper_index] * (underbids[reviewer_index] / sum(underbids)))
    third_step_allocation = copy.deepcopy(fractional_allocation_profile)
    return (first_step_allocation, second_step_allocation, third_step_allocation)

''' example from pdf:
bid_p = [[1, 0, 1, 0, 0, 0],
         [1, 1, 1, 1, 0, 0],
         [0, 1, 1, 0, 1, 0],
         [0, 1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0, 0]]

p = Instance(None, 6, 5, 2)
res = fractional_allocation_algorithm(bid_p, p)
print(res[2])'''
