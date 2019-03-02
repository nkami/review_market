import argparse
import pathlib
import datetime
import csv
import numpy as np
from matching_algorithms import *
from bidder_behaviors import *
from prices_mechanisms import *
from instance_generator import *
from evaluation import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ReviewsAmount", help="sets the number of reviews each paper requires", type=int)
    parser.add_argument("PreflibFile", help="gets the path to the Preflib file input")
    parser.add_argument("Mechanism", help="sets the price mechanism of the simulation", choices=['FixedMechanism',
                                                                                                 'PriceMechanism'])
    args = parser.parse_args()

    instance_generator = PreflibInstanceGenerator(args.PreflibFile, args.ReviewsAmount, SimplePrivatePricesGenerator())
    instance = instance_generator.generate_instance()
    if args.Mechanism == "FixedMechanism":
        price_mec = FixedMechanism(instance)
    elif args.Mechanism == "PriceMechanism":
        price_mec = PriceMechanism(instance)
    else:
        print('wrong mechanism input')
        exit()
    for i in range(0, instance.total_reviewers):
        behavior = SincereIntegralBehavior()
        price_mec.current_bidding_profile = behavior.apply_reviewer_behavior(instance,
                                                                             price_mec.current_bidding_profile, i,
                                                                             price_mec.threshold, price_mec.prices)
        price_mec.update_prices()
    final_bid = price_mec.current_bidding_profile
    allocation_mats = fractional_allocation_algorithm(final_bid, instance)
    try:
        pathlib.Path('.\\Output').mkdir()
    except FileExistsError:
        pass
    with open('.\\Output\\simulation_{0}.csv'.format(datetime.datetime.now().isoformat().replace(':', '_')),
              mode="w") as csv_file:
        output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['Bidder', 'Paper', 'Private Cost', 'Step 1 Allocation',
                                'Step 2 Allocation', 'Step 3 Allocation', 'Social Cost', 'Total Social Cost',
                                'Fairness', 'Unassigned Papers'])
        total_social_cost = calculate_social_cost(instance, allocation_mats[2])
        for bidder in range(0, instance.total_reviewers):
            for paper in range(0, instance.total_papers):
                paper_cost = [pair[1] for pair in instance.private_prices[bidder] if pair[0] == paper]
                if len(paper_cost) == 0:  # temp fix in case of COI
                    paper_cost.append(0)
                paper_cost = paper_cost[0]
                social_cost = paper_cost * allocation_mats[2][bidder][paper]
                output_writer.writerow([str(bidder), str(paper), str(paper_cost),
                                        str(allocation_mats[0][bidder][paper]), str(allocation_mats[1][bidder][paper]),
                                        str(allocation_mats[2][bidder][paper]), str(social_cost),
                                        str(total_social_cost), 'TBD', 'TBD'])
        csv_file.close()


