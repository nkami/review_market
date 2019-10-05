import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from instance_generator import *
from matching_algorithms import *


class Metric:
    def __init__(self):
        pass

    # Gini and Hoover formulas taken from http://www.nickmattei.net/docs/papers.pdf
    def gini_index(self, u):
        um = np.matrix(u)
        U = np.abs(np.transpose(um) - um)
        return U.sum() / (2*len(u)*um.sum())

    def hoover_index(self, u):
        us = np.sum(u)
        return np.sum(np.abs(u - us/len(u))) / (2*us)


class Plotter:
    def __init__(self):
        pass

    def plot_allocation_vs_bids(self, data_frame, output_path):
        # Note: take average over all samples!
        # excess papers ratio(S) = total_excess_papers / total_papers
        # successful bids ratio(S) = allocated_papers_per_bid
        # bids per PCM(P) = total bids / total_reviewers
        price_sensitive_axis = data_frame['fallback probability'].values
        price_sensitive_axis = np.unique(price_sensitive_axis)
        excess_papers_ratio_data = data_frame[['fallback probability', 'm', 'total_excess_papers']]
        excess_papers_ratio = []
        successful_bids_ratio_data = data_frame[['fallback probability', 'allocated_papers_per_bid']]
        successful_bids_ratio = []
        bids_per_PCM_data = data_frame[['fallback probability', 'n', 'total bids']]
        bids_per_PCM = []
        for current_sensitivity in price_sensitive_axis:
            current_papers_ratio_data = excess_papers_ratio_data.loc[excess_papers_ratio_data['fallback probability'] ==
                                                                     current_sensitivity].values
            current_papers_ratio_data = [sample[2] / sample[1] for sample in current_papers_ratio_data]
            excess_papers_ratio.append(sum(current_papers_ratio_data) / len(current_papers_ratio_data))

            current_bids_data = successful_bids_ratio_data.loc[successful_bids_ratio_data['fallback probability'] ==
                                                               current_sensitivity].values
            current_bids_data = [sample[1] for sample in current_bids_data]
            successful_bids_ratio.append(sum(current_bids_data) / len(current_bids_data))

            current_PCM_data = bids_per_PCM_data.loc[bids_per_PCM_data['fallback probability'] == current_sensitivity].values
            current_PCM_data = [sample[2] / sample[1] for sample in current_PCM_data]
            bids_per_PCM.append(sum(current_PCM_data) / len(current_PCM_data))

        fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle('Allocation vs. Bids', y=1)
        excess_papers_color = 'green'
        successful_bids_ratio_color = 'blue'
        ax1.set_xlabel('% price-sensitive bidders')
        ax1.set_ylabel('ratio (S)')
        excess_papers_line, = ax1.plot(price_sensitive_axis, excess_papers_ratio, '-o', color=excess_papers_color)
        successful_bids_line, = ax1.plot(price_sensitive_axis, successful_bids_ratio, '--o', color=successful_bids_ratio_color)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        bids_per_PCM_color = 'black'
        ax2.set_ylabel('count (P)')
        PCM_bids_line, = ax2.plot(price_sensitive_axis, bids_per_PCM, '--o', color=bids_per_PCM_color)
        ax2.tick_params(axis='y')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.legend((excess_papers_line, successful_bids_line, PCM_bids_line),
                   ('Excess Papers Ratio (S)', 'Successful Bids Ratio (S)', 'Bids per PCM (P)'),
                   loc='lower center', ncol=3, fancybox=True, shadow=True, prop={'size': 10})
        plt.savefig(output_path + 'allocation vs bids.png', bbox_inches='tight', pad_inches=0.3, dpi=200)

    def plot_allocation_quality(self, data_frame, output_path):
        # Note: take average over all samples!
        # average cost per bidder(P) = average_bidder_cost
        # average cost per uniform bidder(P) = average_fallback_bidder_cost
        # average cost per price-sensitive bidder(P) = average_main_bidder_cost
        # average cost per requested paper(S) = ?????
        print('to do')

    def plot_allocation_fairness(self, data_frame, output_path):
        # Note: take average over all samples!
        # paper bids = gini_paper_bids
        # uniform bidders costs = gini_fallback_bidder_cost
        # price-sensitive bidders costs = gini_main_bidder_cost
        price_sensitive_axis = data_frame['fallback probability'].values
        price_sensitive_axis = np.unique(price_sensitive_axis)
        paper_bids_data = data_frame[['fallback probability', 'gini_paper_bids']]
        paper_bids = []
        uniform_bidders_costs_data = data_frame[['fallback probability', 'gini_fallback_bidder_cost']]
        uniform_bidders_costs = []
        price_sensitive_bidders_costs_data = data_frame[['fallback probability', 'gini_main_bidder_cost']]
        price_sensitive_bidders_costs = []
        for current_sensitivity in price_sensitive_axis:
            current_papers_bids_data = paper_bids_data.loc[paper_bids_data['fallback probability'] ==
                                                           current_sensitivity].values
            current_papers_bids_data = [sample[1] for sample in current_papers_bids_data]
            paper_bids.append(sum(current_papers_bids_data) / len(current_papers_bids_data))

            current_uniform_costs_data = uniform_bidders_costs_data.loc[uniform_bidders_costs_data['fallback probability'] ==
                                                                        current_sensitivity].values
            current_uniform_costs_data = [sample[1] for sample in current_uniform_costs_data]
            uniform_bidders_costs.append(sum(current_uniform_costs_data) / len(current_uniform_costs_data))

            current_sensitive_bidders_costs_data = price_sensitive_bidders_costs_data.loc[
                price_sensitive_bidders_costs_data['fallback probability'] == current_sensitivity].values
            current_sensitive_bidders_costs_data = [sample[1] for sample in current_sensitive_bidders_costs_data]
            price_sensitive_bidders_costs.append(sum(current_sensitive_bidders_costs_data) / len(current_sensitive_bidders_costs_data))

        fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle('Allocation Fairness', y=1)
        paper_bids_color = 'black'
        uniform_bidders_costs_color = 'orange'
        price_sensitive_bidders_costs_color = 'red'
        ax1.set_xlabel('% price-sensitive bidders')
        ax1.set_ylabel('Gini Coefficient')
        paper_bids_line, = ax1.plot(price_sensitive_axis, paper_bids, '-o', color=paper_bids_color)
        uniform_bidders_costs_line, = ax1.plot(price_sensitive_axis, uniform_bidders_costs, '--o',
                                               color=uniform_bidders_costs_color)
        price_sensitive_bidders_costs_line, = ax1.plot(price_sensitive_axis, price_sensitive_bidders_costs, '--o',
                                                       color=price_sensitive_bidders_costs_color)
        fig.tight_layout()
        fig.legend((paper_bids_line, uniform_bidders_costs_line, price_sensitive_bidders_costs_line),
                   ('Paper Bids', 'Uniform Bidders Costs', 'Price Sensitive Bidders Costs'),
                   loc='lower center', ncol=3, fancybox=True, shadow=True, prop={'size': 10})
        plt.savefig(output_path + 'allocation fairness.png', bbox_inches='tight', pad_inches=0.3, dpi=200)

    def plot_cost_per_PCM_algorithm_dependant(self, data_frame, output_path):
        # Note: take average over all samples!
        # average cost per bidder(P) = average_bidder_cost
        # compliance = fallback probability related?
        print('to do')

    def plot_cost_per_PCM_bidding_rounds_dependant(self, data_frame, output_path):
        # Note: take average over all samples!
        # ???????????????????
        print('to do')

    def plot_cost_per_PCM_alpha_dependant(self, data_frame, output_path):
        # Note: take average over all samples!
        # ???????????????????
        print('to do')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputPath", help="a path to the input file or directory")
    args = parser.parse_args()
    data_frame = pd.read_csv(args.InputPath)
    output_path = args.InputPath.split('/')[:-1]
    output_path = '/'.join(output_path) + '/'
    plotter = Plotter()
    plotter.plot_allocation_vs_bids(data_frame, output_path)
    plotter.plot_allocation_fairness(data_frame, output_path)

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
