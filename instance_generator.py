import random


class Instance:
    '''
    A problem instance which consists of a preferences profile, total amount of papers, total amount of reviewers and
    papers review requirements. Papers review requirements is a list that assigns each paper the amount of times it
    should be reviewed. (the default is to set all the papers requirement to the same constant: r)
    '''
    def __init__(self, instance_private_prices, number_of_papers, number_of_reviewers, papers_requirements):
        self.private_prices = instance_private_prices
        self.total_papers = number_of_papers
        self.total_reviewers = number_of_reviewers
        if isinstance(papers_requirements, int):
            requirements_list = []
            for i in range(0, self.total_papers):
                requirements_list.append(papers_requirements)
            self.papers_review_requirement = requirements_list
        else:
            self.papers_review_requirement = papers_requirements


class PrivatePricesGenerator:
    '''
    In every Instance, bidders assign private prices to each paper according to the preferences acquired from the
    problem instance (e.g from a Preflib file or some other method)
    '''
    def generate_private_prices(self, preferences):
        print('Method not implemented')


class SimplePrivatePricesGenerator(PrivatePricesGenerator):
    '''
    Assign a random private cost to each paper, while satsifying the condition that every paper with preference
    rank i > j will have lower cost than every paper with preference rank j (i.e. papers that are more desirable
    according to preference will have a lower cost)
    '''
    def generate_private_prices(self, preferences):
        private_prices = []
        for reviewer in range(0, len(preferences)):
            reviewer_prices = []
            for rank, reviewer_preferences in enumerate(preferences[reviewer]):
                if -1 in reviewer_preferences:  # preference is empty
                    continue
                for paper in reviewer_preferences:
                    reviewer_prices.append((paper, random.uniform(rank, rank + 1)))
            private_prices.append(reviewer_prices)
        return private_prices


class InstanceGenerator:
    def __init__(self, papers_requirements, private_prices_generator):
        self.number_of_papers = None
        self.private_prices_generator = private_prices_generator
        self.papers_review_requirement = papers_requirements

    '''
    Returns a preferences profile, which is a list of lists of tuples. each list is a reviewer profile,
    that contains tuples of papers indexes. The tuples in each profile are sorted in a descending order of reviewer
    preference rank (first tuple is the most desirable).
    '''
    def create_preferences(self):
        print('Method not implemented')

    '''
    Returns a problem instance.
    '''
    def generate_instance(self):
        instance_preferences = self.create_preferences()
        private_prices = self.private_prices_generator.generate_private_prices(instance_preferences)
        number_of_reviewers = len(instance_preferences)
        number_of_papers = self.number_of_papers
        return Instance(private_prices, number_of_papers, number_of_reviewers, self.papers_review_requirement)


class PreflibInstanceGenerator(InstanceGenerator):
    def __init__(self, input_file_path, papers_requirements, private_prices_generator):
        super().__init__(papers_requirements, private_prices_generator)
        self.file = open(input_file_path)

    '''
    Returns a preferences profile from a Preflib file.
    '''
    def create_preferences(self):
        instance_preferences = []
        starting_line = int(self.file.readline())
        self.number_of_papers = starting_line
        for line_number, line in enumerate(self.file.readlines()):
            if line_number > starting_line:
                reviewer_profile = []
                modified_line = ""
                for i in range(0, len(line)):
                    if i > 2 and i < (len(line) - 3):
                        modified_line += line[i]
                for string_ranked_preference in modified_line.split('},{'):
                    if string_ranked_preference == '':  # in case a preference rank is empty assign dummy paper -1
                        string_ranked_preference = '-1'
                    ranked_preference = tuple(map(int, string_ranked_preference.split(',')))
                    reviewer_profile.append(ranked_preference)
                instance_preferences.append(reviewer_profile)
        return instance_preferences


