import random
import json
import argparse
import numpy as np
import pathlib
import datetime


class Instance:
    # A problem instance which consists of a preferences profile, total amount of papers, total amount of reviewers and
    # papers review requirements. Papers review requirements is a list that assigns each paper the amount of times it
    # should be reviewed. (the default is to set all the papers requirement to the same constant: r)
    def __init__(self, instance_coi, instance_private_prices, number_of_papers, number_of_reviewers,
                 papers_requirements, unallocated_papers_price, instance_preferences):
        self.private_prices = instance_private_prices
        self.total_papers = number_of_papers
        self.total_reviewers = number_of_reviewers
        self.coi = instance_coi
        self.preferences_profile = instance_preferences
        if isinstance(papers_requirements, int):
            self.papers_review_requirement = [papers_requirements for x in range(0, number_of_papers)]
        else:
            self.papers_review_requirement = papers_requirements
        if isinstance(unallocated_papers_price, int):
            self.unallocated_papers_price = [unallocated_papers_price for x in range(0, number_of_papers)]
        else:
            self.unallocated_papers_price = unallocated_papers_price


class PrivatePricesGenerator:
    def __init__(self, params):
        pass

    # In every Instance, bidders assign private prices to each paper according to the preferences acquired from the
    # problem instance (e.g from a Preflib file or some other method)
    def generate_private_prices(self, preferences):
        print('Method not implemented')


class SimplePrivatePricesGenerator(PrivatePricesGenerator):
    # Assign a random private cost to each paper, while satisfying the condition that every paper with preference
    # rank i > j will have lower cost than every paper with preference rank j (i.e. papers that are more desirable
    # according to preference will have a lower cost)
    def generate_private_prices(self, preferences):
        private_prices = []
        for reviewer in range(0, len(preferences)):
            reviewer_prices = {}
            for rank, reviewer_preferences in enumerate(preferences[reviewer]):
                if -1 in reviewer_preferences:  # preference is empty
                    continue
                for paper in reviewer_preferences:
                    reviewer_prices[str(paper)] = random.uniform(rank, rank + 1)
            private_prices.append(reviewer_prices)
        return private_prices


class InstanceGenerator:
    def __init__(self, params):
        self.number_of_papers = None
        self.private_prices_generator = possible_private_price_generators[params['private_prices_generator']](params)
        self.papers_review_requirement = params['papers_requirements']
        self.unallocated_papers_price = params['unallocated_papers_price']

    # Returns a preferences profile and a COI list, a preferences profile is a list of lists of tuples. each list is a
    # reviewer profile, that contains tuples of papers indices. The tuples in each profile are sorted in a descending
    # order of reviewer preference rank (first tuple is the most desirable). A COI list is a list of lists that contains
    # the papers indices that each reviewer has a conflict of interest with.
    def create_pref_and_coi(self):
        print('Method not implemented')

    # Returns a problem instance.
    def generate_instance(self):
        instance_pref_and_coi = self.create_pref_and_coi()
        instance_preferences = [pair[0] for pair in instance_pref_and_coi]
        instance_coi = [pair[1] for pair in instance_pref_and_coi]
        private_prices = self.private_prices_generator.generate_private_prices(instance_preferences)
        number_of_reviewers = len(instance_preferences)
        number_of_papers = self.number_of_papers
        return Instance(instance_coi, private_prices, number_of_papers, number_of_reviewers,
                        self.papers_review_requirement, self.unallocated_papers_price, instance_preferences)


class PreflibInstanceGenerator(InstanceGenerator):
    def __init__(self, params):
        super().__init__(params)
        self.file = open(params['additional_params']['PreflibFile'])

    # Returns a preferences profile and a COI list from a Preflib file.
    def create_pref_and_coi(self):
        instance_pref_and_coi = []
        starting_line = int(self.file.readline())
        self.number_of_papers = starting_line
        for line_number, line in enumerate(self.file.readlines()):
            if line_number > starting_line:
                reviewer_profile = []
                reviewer_no_coi = []
                modified_line = ""
                if line[-1:] != ',':
                    line = line + ','
                for i in range(0, len(line)):
                    if i > 2 and i < (len(line) - 3):
                        modified_line += line[i]
                for string_ranked_preference in modified_line.split('},{'):
                    if string_ranked_preference == '':  # in case a preference rank is empty assign dummy paper -1
                        string_ranked_preference = '-1'
                    ranked_preference = tuple(map(int, string_ranked_preference.split(',')))
                    rank_papers = list(ranked_preference)
                    reviewer_no_coi = reviewer_no_coi + rank_papers
                    reviewer_profile.append(ranked_preference)
                reviewer_coi = [paper for paper in range(0, self.number_of_papers) if paper not in reviewer_no_coi]
                number_of_identical_preferences = int(line[0])
                for _ in range(0, number_of_identical_preferences):
                    instance_pref_and_coi.append((reviewer_profile, reviewer_coi))
        return instance_pref_and_coi


possible_instance_generators = {'PreflibInstanceGenerator': PreflibInstanceGenerator}
possible_private_price_generators = {'SimplePrivatePricesGenerator': SimplePrivatePricesGenerator}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputFile", help="a path to the input file")
    args = parser.parse_args()
    with open(args.InputFile) as file:
        params = json.loads(file.read())
    instance_generator = possible_instance_generators[params['instance_generator']](params)
    instance = instance_generator.generate_instance()
    quota_matrix = np.ones((instance.total_reviewers, instance.total_papers))
    for reviewer in range(0, instance.total_reviewers):
        for paper in range(0, instance.total_papers):
            if params['ignore_quota_constraints'] == True:
                quota_matrix[reviewer][paper] = np.inf
            elif paper in instance.coi[reviewer]:
                quota_matrix[reviewer][paper] = 0
    cost_matrix = np.zeros((instance.total_reviewers, instance.total_papers))
    for reviewer in range(0, instance.total_reviewers):
        for paper in range(0, instance.total_papers):
            if str(paper) in instance.private_prices[reviewer].keys():  # a coi paper will not have a price
                cost_matrix[reviewer][paper] = instance.private_prices[reviewer][str(paper)]
    output = {'reviewers_behavior': 'fill',
              'forced_permutations': 'fill',
              'number_of_bids_until_prices_update': 'fill',
              'total_bids_until_closure': 'fill',
              'matching_algorithm': 'fill',
              'market_mechanism': 'fill',
              'ignore_quota_constraints': params['ignore_quota_constraints'],
              'additional_params': 'fill',
              'total_reviewers': instance.total_reviewers,
              'total_papers': instance.total_papers,
              'papers_requirements': instance.papers_review_requirement,
              'unallocated_papers_price': instance.unallocated_papers_price,
              'cost_matrix': cost_matrix.tolist(),
              'quota_matrix': quota_matrix.tolist()}
    try:
        pathlib.Path('.\\output').mkdir()
    except FileExistsError:
        pass
    path = '.\\output\\cost_matrix_{0}.json'.format(datetime.datetime.now().isoformat()[:-7].replace(':', '-'))
    with open(path, 'w') as output_file:
        json.dump(output, output_file, indent=4)
