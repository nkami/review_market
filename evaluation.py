from instance_generator import *
from matching_algorithms import *


# Gini and Hoover formulas taken from http://www.nickmattei.net/docs/papers.pdf
def gini_index(u):
    um = np.matrix(u)
    U = np.abs(np.transpose(um) - um)
    return U.sum() / (2*len(u)*um.sum())


def hoover_index(u):
    us = np.sum(u)
    return np.sum(np.abs(u - us/len(u))) / (2*us)


def plot_allocation_vs_bids():
    # Note: take average over all samples!
    # excess papers ratio(S) = total_excess_papers / total_papers
    # successful bids ratio(S) = allocated_papers_per_bid
    # bids per PCM(P) = total_bids / total_reviewers
    print('to do')


def plot_allocation_quality():
    # Note: take average over all samples!
    # average cost per bidder(P) = average_bidder_cost
    # average cost per uniform bidder(P) = average_fallback_bidder_cost
    # average cost per price-sensitive bidder(P) = average_main_bidder_cost
    # average cost per requested paper(S) = ?????
    print('to do')


def plot_allocation_fairness():
    # Note: take average over all samples!
    # paper bids = gini_paper_bids
    # uniform bidders costs = gini_fallback_bidder_cost
    # price-sensitive bidders costs = gini_main_bidder_cost
    print('to do')


def plot_cost_per_PCM_algorithm_dependant():
    # Note: take average over all samples!
    # average cost per bidder(P) = average_bidder_cost
    # compliance = fallback probability related?
    print('to do')


def plot_cost_per_PCM_bidding_rounds_dependant():
    # Note: take average over all samples!
    # ???????????????????
    print('to do')


def plot_cost_per_PCM_alpha_dependant():
    # Note: take average over all samples!
    # ???????????????????
    print('to do')





# bidding_p = [[5, 5, 1.5, 5, 5, 5],
#              [5, 1.2, 5, 0.6, 5, 5],
#              [1, 1, 5, 5, 5, 5]]
# params = {}
# params['total_papers'] = 6
# params['total_reviewers'] = 3
# params['quota_matrix'] = [[1, 1, 1, 1, 1, 1],
#                           [1, 1, 1, 1, 1, 1],
#                           [1, 1, 1, 1, 1, 1]]
# params['papers_requirements'] = [2.5 for i in range(0, params['total_papers'])]
# algo = FractionalSumOWA(params)
# res = algo.match(bidding_p, params)
# print(res['third_step_allocation'])
