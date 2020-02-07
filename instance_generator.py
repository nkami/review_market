import random
import json
import argparse
import numpy as np
import pathlib
import datetime
import re

rank_cost_lower_bounds = [0, 1, 2, 8, 15]
rank_cost_upper_bounds = [1, 2, 8, 15, 20]
matrices_local_dir = ".\\cost_matrices"


class Instance:
    # A problem instance which consists of a preferences profile, total amount of papers, total amount of reviewers and
    # papers review requirements. Papers review requirements is a list that assigns each paper the amount of times it
    # should be reviewed. (the default is to set all the papers requirement to the same constant: r)
    def __init__(self, instance_coi, instance_private_costs, number_of_papers, number_of_reviewers,
                 papers_requirements,  instance_preferences):
        self.private_costs = instance_private_costs
        self.total_papers = number_of_papers
        self.total_reviewers = number_of_reviewers
        self.coi = instance_coi
        self.preferences_profile = instance_preferences
        if isinstance(papers_requirements, int):
            self.papers_review_requirement = [papers_requirements for x in range(0, number_of_papers)]
        else:
            self.papers_review_requirement = papers_requirements
        # if isinstance(unallocated_papers_cost, int):
        #     self.unallocated_papers_cost = [unallocated_papers_cost for x in range(0, number_of_papers)]
        # else:
        #     self.unallocated_papers_cost = unallocated_papers_cost


class PrivatePricesGenerator:
    def __init__(self, params):
        pass

    # In every Instance, bidders assign private prices to each paper according to the preferences acquired from the
    # problem instance (e.g from a Preflib file or some other method)
    def generate_private_costs(self, preferences):
        print('Method not implemented')


class SimplePrivatePricesGenerator(PrivatePricesGenerator):
    # Assign a random private cost to each paper, while satisfying the condition that every paper with preference
    # rank i > j will have lower cost than every paper with preference rank j (i.e. papers that are more desirable
    # according to preference will have a lower cost)
    def generate_private_costs(self, preferences):
        private_costs = []
        for reviewer in range(0, len(preferences)):
            reviewer_costs = {}
            for rank, reviewer_preferences in enumerate(preferences[reviewer]):
                if -1 in reviewer_preferences:  # preference is empty
                    continue
                for paper in reviewer_preferences:
                    reviewer_costs[str(paper)] = random.uniform(rank_cost_lower_bounds[rank], rank_cost_upper_bounds[rank])
            private_costs.append(reviewer_costs)
        return private_costs


class InstanceGenerator:
    def __init__(self, params):
        self.number_of_papers = None
        self.number_of_reviewers = None
        self.private_prices_generator = possible_private_price_generators[params['private_prices_generator']](params)
        self.papers_review_requirement = params['papers_requirements']


    # Returns a preferences profile and a COI list, a preferences profile is a list of lists of tuples. each list is a
    # reviewer profile, that contains tuples of papers indices. The tuples in each profile are sorted in a descending
    # order of reviewer preference rank (first tuple is the most desirable). A COI list is a list of lists that contains
    # the papers indices that each reviewer has a conflict of interest with.
    def create_pref_and_coi(self):
        print('Method not implemented')

    # process the bid list. Should be overwritten in children classes
    def process(self,bid_list):
        return bid_list

    # Returns a problem instance.
    def generate_instance(self):
        instance_pref_and_coi = self.create_pref_and_coi()
        self.number_of_reviewers = len(instance_pref_and_coi)
        instance_pref_and_coi = self.process(instance_pref_and_coi)
        instance_preferences = [pair[0] for pair in instance_pref_and_coi]
        instance_coi = [pair[1] for pair in instance_pref_and_coi]
        private_prices = self.private_prices_generator.generate_private_costs(instance_preferences)
        number_of_reviewers = self.number_of_reviewers
        number_of_papers = self.number_of_papers
        return Instance(instance_coi, private_prices, number_of_papers, number_of_reviewers,
                        self.papers_review_requirement,  instance_preferences)


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
            line = re.sub('[\n]', '', line)
            line = re.sub('},{', ';', line)
            line = re.sub(',{', ';', line)
            line = re.sub('},', '', line)
            line = re.sub('}', '', line)
            if line_number > starting_line:
                reviewer_profile = []
                reviewer_no_coi = []
                for string_ranked_preference in line.split(';')[1:]:
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

class PreflibSampleInstanceGenerator(PreflibInstanceGenerator):
    def __init__(self, params):
        super().__init__(params)
        self.file = open(params['additional_params']['PreflibFile'])
        self.sample_size_n = params['additional_params']['sample_size_n']
        self.sample_size_m = params['additional_params']['sample_size_m']

    # sample n bidders and m papers, renames papers as 0..m-1, and removes all other papers and bidders
    def process(self,bid_list):
        sample_n = random.sample(range(self.number_of_reviewers),self.sample_size_n)
        sample_m = random.sample(range(self.number_of_papers),self.sample_size_m)
        bid_list = list(bid_list[x] for x in sample_n)
        new_paper_indices = [np.nan]*(self.number_of_papers+1)
        for ind,old_ind in enumerate(sample_m):
            new_paper_indices[old_ind+1] = ind
        new_paper_indices[0] = -1   #  reserved for the "-1" values in the lists
        for i in range(self.sample_size_n):
            ranks = len(bid_list[i][0])
            for rank in range(ranks):
                bid_list[i][0][rank] = tuple(new_paper_indices[x+1] for x in bid_list[i][0][rank] if x in sample_m or x == -1)
            if len(bid_list[i][1])>0:
                for ind,x in enumerate(bid_list[i][1]):
                    if x in sample_m:
                        bid_list[i][1][ind] = new_paper_indices[x+1]
                    else:
                        bid_list[i][1].remove(x)
            self.number_of_reviewers = self.sample_size_n
            self.number_of_papers = self.sample_size_m
        return bid_list

possible_instance_generators = {'PreflibInstanceGenerator': PreflibInstanceGenerator,
                                'PreflibSampleInstanceGenerator': PreflibSampleInstanceGenerator}
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
            if str(paper) in instance.private_costs[reviewer].keys():  # a coi paper will not have a price
                cost_matrix[reviewer][paper] = instance.private_costs[reviewer][str(paper)]
    cost_matrix_output = {'preflib_file': params['additional_params']['PreflibFile'],
                          'total_reviewers': instance.total_reviewers,
                          'total_papers': instance.total_papers,
                          'cost_matrix': cost_matrix.tolist()}
    quota_matrix_output = {'preflib_file': params['additional_params']['PreflibFile'],
                           'total_reviewers': instance.total_reviewers,
                           'total_papers': instance.total_papers,
                           'quota_matrix': quota_matrix.tolist()}
    try:
        pathlib.Path(matrices_local_dir).mkdir()
    except FileExistsError:
        pass
    time_stamp = datetime.datetime.now().isoformat()[:-7].replace(':', '-')
    if "file_suffix" in params:
        file_suffix = params["file_suffix"]
    else:
        file_suffix = time_stamp
    cost_matrix_path = '{1}\\cost_matrix_m{2}_n{3}_{0}.json'.format(file_suffix, matrices_local_dir,instance.total_papers,instance.total_reviewers)
    quota_matrix_path = '{1}\\quota_matrix_m{2}_n{3}_{0}.json'.format(file_suffix, matrices_local_dir,instance.total_papers,instance.total_reviewers)
    data_and_paths = [(cost_matrix_path, cost_matrix_output), (quota_matrix_path, quota_matrix_output)]
    for current_pair in data_and_paths:
        with open(current_pair[0], 'w') as output_file:
            json.dump(current_pair[1], output_file, indent=4)
